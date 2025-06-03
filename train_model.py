import warnings
warnings.filterwarnings("ignore")
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset
import torch
import os
import json
import pandas as pd
import pyodbc
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import threading

# Configure logging
log_file = "train_codes_1b_log.txt"
status_file = "training_status.json"

try:
    with open(log_file, "w") as f:
        f.write("")
    with open(status_file, "w") as f:
        json.dump({}, f)
except Exception as e:
    print(f"Error accessing log or status files: {e}")
    raise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logging.getLogger("transformers").addHandler(logging.FileHandler(log_file))
logging.getLogger("transformers").addHandler(logging.StreamHandler())
logging.getLogger("transformers").setLevel(logging.INFO)

# Callback for logging training progress
class ProgressCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("Training started")
        update_status("training", {"status": "running", "step": 0, "timestamp": datetime.now().isoformat()})

    def on_step_end(self, args, state, control, **kwargs):
        progress = (state.global_step / state.max_steps) * 100
        loss = state.log_history[-1].get("loss", "N/A") if state.log_history else "N/A"
        logger.info(f"Step {state.global_step}/{state.max_steps}, loss: {loss}, Progress: {progress:.1f}%")
        update_status("training", {"status": "running", "step": state.global_step, "timestamp": datetime.now().isoformat()})

    def on_train_end(self, args, state, control, **kwargs):
        logger.info("Training completed")
        update_status("training", {"status": "finished", "step": state.global_step, "timestamp": datetime.now().isoformat()})

    def on_error(self, args, state, control, **kwargs):
        logger.info(f"Training interrupted at step {state.global_step}")
        update_status("training", {"status": "error", "step": state.global_step, "timestamp": datetime.now().isoformat()})

# Function to update training status
def update_status(process_type, status_data):
    try:
        status = {}
        if os.path.exists(status_file):
            with open(status_file, "r") as f:
                status = json.load(f)
        status[process_type] = status_data
        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        logger.error(f"Error updating status: {e}")
        print(f"Error updating status: {e}")

# Model and training parameters
model_name = "seeklhy/codes-1b"
output_dir = "./codes_1b_sql_generator"
data_file = "training_data.csv"
max_input_length = 128
max_new_tokens = 256
num_epochs = 3
batch_size = 4
gradient_accumulation_steps = 8
learning_rate = 2e-5
weight_decay = 0.01

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if device.type == "cpu":
    warnings.warn("GPU not detected. Training on CPU will be very slow.")

if device.type == "cuda":
    torch.cuda.empty_cache()

# Function to fetch data from Azure SQL
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
def fetch_data_from_azure():
    try:
        connection_string = (
            "Driver={ODBC Driver 17 for SQL Server};"
            "Server=tcp:nl-two-sql-server.database.windows.net,1433;"
            "Database=nl_sql_db;"
            "UID=sqladmin;"
            "PWD=EvgeneMaks2002;"
            "Encrypt=yes;"
            "TrustServerCertificate=no;"
            "Connection Timeout=30;"
        )
        conn = pyodbc.connect(connection_string)
        query = "SELECT NaturalLanguageQuery, GeneratedSql FROM SqlTrainingData WHERE NaturalLanguageQuery IS NOT NULL AND GeneratedSql IS NOT NULL"
        df = pd.read_sql(query, conn)
        conn.close()
        
        df = df.rename(columns={"NaturalLanguageQuery": "Text", "GeneratedSql": "GeneratedSql"})
        df.to_csv(data_file, index=False)
        return df
    except Exception as e:
        logger.error(f"Error connecting to Azure SQL: {e}")
        raise e

# Function to generate SQL from natural language
def generate_sql(text, model, tokenizer, max_input_length=128, max_new_tokens=256, device="cuda"):
    try:
        model.eval()
        # Input prompt format for causal language model
        input_prompt = f"### Instruction: Translate the following natural language query to SQL:\n\n{text}\n\n### SQL Query:\n"
        inputs = tokenizer(
            input_prompt,
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True,
            padding=True
        )
        inputs = inputs.to(device)
        
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=3,
            temperature=0.7,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Decode only the SQL part, removing the prompt
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        sql_start = generated_text.find("### SQL Query:\n")
        if sql_start != -1:
            sql_text = generated_text[sql_start + len("### SQL Query:\n"):].strip()
            return sql_text
        else:
            return generated_text.strip()
    except Exception as e:
        logger.error(f"Error generating SQL: {str(e)}")
        return f"Error generating SQL: {str(e)}"

# Function to train the model
def train_model():
    try:
        if os.path.exists(output_dir):
            logger.info(f"Directory {output_dir} already exists")
            return {"status": "error", "message": f"Directory {output_dir} already exists. Training aborted."}

        logger.info("Loading data...")
        df = fetch_data_from_azure()
        logger.info(f"Loaded {len(df)} records from Azure SQL")

        if len(df) < 2000 or len(df) > 3000:
            logger.info(f"Number of records ({len(df)}) is not within the 2000–3000 range")
            return {"status": "error", "message": "Number of records must be between 2000 and 3000"}

        dataset = Dataset.from_pandas(df)
        logger.info(f"Dataset size: {len(dataset)} examples")
        
        logger.info(f"Loading model {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)

        # Set pad_token if not defined
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        # Preprocess dataset
        def preprocess_function(examples):
            inputs = [f"### Instruction: Translate the following natural language query to SQL:\n\n{text}\n\n### SQL Query:\n{sql}" for text, sql in zip(examples["Text"], examples["GeneratedSql"])]
            model_inputs = tokenizer(
                inputs,
                max_length=max_input_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            model_inputs["labels"] = model_inputs["input_ids"].clone()
            return model_inputs

        logger.info("Preprocessing dataset...")
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=["Text", "GeneratedSql"],
            load_from_cache_file=False
        )
        
        logger.info("Splitting dataset...")
        split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
        logger.info(f"Training set size: {len(split_dataset['train'])}, Test set size: {len(split_dataset['test'])}")

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,                    
            num_train_epochs=num_epochs,             
            per_device_train_batch_size=batch_size,  
            per_device_eval_batch_size=batch_size,    
            gradient_accumulation_steps=gradient_accumulation_steps,  
            learning_rate=learning_rate,             
            weight_decay=weight_decay,              
            logging_steps=50,                        
            save_steps=200,                           
            save_total_limit=2,                      
            eval_strategy="steps",                   
            eval_steps=200,                         
            load_best_model_at_end=True,            
            metric_for_best_model="loss",            
            greater_is_better=False,                  
            fp16=(device.type == "cuda"),             
            gradient_checkpointing=True,              
            report_to="none"                         
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
            callbacks=[ProgressCallback()]
        )

        logger.info("Starting training...")
        update_status("training", {"status": "running", "step": 0, "timestamp": datetime.now().isoformat()})
        
        trainer.train()
        
        logger.info("Training completed")
        update_status("training", {"status": "Finished", "training_step": trainer.state.max_steps, "timestamp": datetime.now().isoformat()})

        logger.info("Saving model...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info("Testing model on sample queries...")
        test_queries = [
            "Give me name organizations",
            "Give me tickets",
            "Give me all deal organizations"
        ]

        for query in test_queries:
            generated_sql = generate_sql(query, model, tokenizer, max_input_length, max_new_tokens, device)
            logger.info(f"Test query: {query}")
            logger.info(f"Generated SQL: {generated_sql}")
            logger.info("---")

        logger.info("Model testing completed")
        return {"status": "success", "message": "Training completed successfully"}

    except Exception as e:
        logger.error(f"Training error: {e}")
        update_status("training", {"status": "error", "step": -1, "timestamp": datetime.now().isoformat()})
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(data_file):
            try:
                os.remove(data_file)
                logger.info("File data.csv deleted")
            except Exception as e:
                logger.error(f"Error deleting {data_file}: {e}")

def start_training():
    if os.path.exists(output_dir):
        return {"status": "error", "message": f"Directory {output_dir} already exists. Training aborted."}
    
    def run_training():
        result = train_model()
        logger.info(f"Training result: {result}")
    
    training_thread = threading.Thread(target=run_training)
    training_thread.start()
    return {"status": "success", "message": "Training started in the background"}

if __name__ == "__main__":
    start_training()