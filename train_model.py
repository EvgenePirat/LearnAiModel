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
import traceback

# Configure logging
log_file = "train_codes_1b_log.txt"
status_file = "training_status.json"

# Initialize logger early to capture all errors
logger = logging.getLogger(__name__)

# Check directory write permissions
try:
    log_dir = os.path.dirname(log_file) or "."
    logger.info(f"Checking write access to directory: {log_dir}")
    if not os.access(log_dir, os.W_OK):
        raise PermissionError(f"No write permission in directory: {log_dir}")
except Exception as e:
    print(f"Fatal error checking directory access: {str(e)}")
    raise

# Initialize log and status files
try:
    logger.info(f"Creating or clearing log file: {log_file}")
    with open(log_file, "w") as f:
        f.write("")
    logger.info(f"Creating or clearing status file: {status_file}")
    with open(status_file, "w") as f:
        json.dump({}, f)
except Exception as e:
    print(f"Error accessing log or status files: {str(e)}")
    logger.error(f"Error accessing log or status files: {str(e)}")
    raise

# Configure logging handlers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a"),
        logging.StreamHandler()
    ]
)

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
        logger.error(f"Training interrupted at step {state.global_step}")
        update_status("training", {"status": "error", "step": state.global_step, "timestamp": datetime.now().isoformat()})

# Function to update training status
def update_status(process_type, status_data):
    try:
        logger.info(f"Updating status for {process_type}: {status_data}")
        status = {}
        if os.path.exists(status_file):
            logger.info(f"Reading existing status file: {status_file}")
            with open(status_file, "r") as f:
                status = json.load(f)
        status[process_type] = status_data
        logger.info(f"Writing updated status to {status_file}")
        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)
        logger.info(f"Status updated successfully")
    except Exception as e:
        logger.error(f"Error updating status: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        print(f"Error updating status: {str(e)}")

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
        logger.info("Attempting to connect to Azure SQL")
        connection_string = (
            "Driver={ODBC Driver 17 for SQL Server};"
            "Server=tcp:nl-two-sql-server.database.windows.net,1433;"
            "Database=nl_sql_db;"
            "UID=sqladmin;"
            "PWD=EvgeneMaks2002;"
            "Encrypt=yes;"
            "TrustServerCertificate=no;"
            "Connection Timeout=60;"
        )
        logger.info("Connecting to Azure SQL with provided connection string")
        conn = pyodbc.connect(connection_string)
        logger.info("Connected to Azure SQL successfully")
        query = "SELECT TOP 3000 NaturalLanguageQuery, GeneratedSql FROM SqlTrainingData WHERE NaturalLanguageQuery IS NOT NULL AND GeneratedSql IS NOT NULL"
        logger.info(f"Executing query: {query}")
        df = pd.read_sql(query, conn)
        logger.info(f"Fetched {len(df)} records from Azure SQL")
        conn.close()
        
        df = df.rename(columns={"NaturalLanguageQuery": "Text", "GeneratedSql": "GeneratedSql"})
        logger.info(f"Saving data to {data_file}")
        df.to_csv(data_file, index=False)
        logger.info(f"Data saved successfully to {data_file}")
        return df
    except Exception as e:
        logger.error(f"Error connecting to Azure SQL: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise e

# Function to train the model
def train_model():
    try:
        logger.info("Starting train_model function")
        if os.path.exists(output_dir):
            logger.info(f"Directory {output_dir} already exists")
            return {"status": "error", "message": f"Directory {output_dir} already exists. Training aborted."}

        logger.info("Loading data from Azure SQL")
        df = fetch_data_from_azure()
        logger.info(f"Loaded {len(df)} records from Azure SQL")

        dataset = Dataset.from_pandas(df)
        logger.info(f"Dataset size: {len(dataset)} examples")
        
        logger.info(f"Loading model {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Tokenizer loaded successfully")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        logger.info("Model loaded successfully")
        model.to(device)
        logger.info(f"Model moved to {device}")

        # Set pad_token if not defined
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
            logger.info("Pad token set to EOS token")

        # Preprocess dataset
        def preprocess_function(examples):
            logger.info("Preprocessing dataset batch")
            inputs = [f"### Instruction: Translate the following natural language query to SQL:\n\n{text}\n\n### SQL Query:\n{sql}" for text, sql in zip(examples["Text"], examples["GeneratedSql"])]
            model_inputs = tokenizer(
                inputs,
                max_length=max_input_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            model_inputs["labels"] = model_inputs["input_ids"].clone()
            logger.info("Batch preprocessing completed")
            return model_inputs

        logger.info("Starting dataset preprocessing")
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=["Text", "GeneratedSql"],
            load_from_cache_file=False
        )
        logger.info("Dataset preprocessing completed")
        
        logger.info("Splitting dataset")
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
        logger.info("Training arguments defined")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
            callbacks=[ProgressCallback()]
        )
        logger.info("Trainer initialized")

        logger.info("Starting training...")
        update_status("training", {"status": "running", "step": 0, "timestamp": datetime.now().isoformat()})
        
        trainer.train()
        logger.info("Training completed")
        update_status("training", {"status": "finished", "training_step": trainer.state.max_steps, "timestamp": datetime.now().isoformat()})

        logger.info("Saving model...")
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved to {output_dir}")

        logger.info("Model training and saving completed")
        return {"status": "success", "message": "Training completed successfully"}

    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        update_status("training", {"status": "error", "step": -1, "timestamp": datetime.now().isoformat()})
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(data_file):
            try:
                os.remove(data_file)
                logger.info(f"File {data_file} deleted")
            except Exception as e:
                logger.error(f"Error deleting {data_file}: {str(e)}")

def start_training():
    try:
        logger.info("Starting training process")
        if os.path.exists(output_dir):
            logger.info(f"Directory {output_dir} already exists")
            return {"status": "error", "message": f"Directory {output_dir} already exists. Training aborted."}
        
        def run_training():
            try:
                logger.info("Training thread started")
                result = train_model()
                logger.info(f"Training result: {result}")
            except Exception as e:
                logger.error(f"Error in training thread: {str(e)}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
        
        logger.info("Creating training thread")
        training_thread = threading.Thread(target=run_training)
        training_thread.start()
        logger.info("Training thread started successfully")
        return {"status": "success", "message": "Training started in the background"}
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    logger.info("Running train_model.py directly")
    start_training()