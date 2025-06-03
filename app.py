import warnings
warnings.filterwarnings("ignore")
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
import logging
from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
from flask_cors import CORS
from datetime import datetime
from train_model import start_training  # Import start_training from train_model.py

# Configure logging
log_file = "train_codes_1b_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logging.getLogger("transformers").addHandler(logging.FileHandler(log_file))
logging.getLogger("transformers").addHandler(logging.StreamHandler())
logging.getLogger("transformers").setLevel(logging.INFO)

# Model parameters
output_dir = "./codes_1b_sql_generator"
max_input_length = 128
max_new_tokens = 256

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if device.type == "cpu":
    warnings.warn("GPU not detected. Using CPU.")

if device.type == "cuda":
    torch.cuda.empty_cache()

# Function to update training status
def update_status(process_type, status_data):
    try:
        status_file = "training_status.json"
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

# Function to generate SQL from natural language
def generate_sql(text, model, tokenizer, max_input_length=128, max_new_tokens=256, device="cuda"):
    try:
        model.eval()
        input_prompt = f"### Instruction: Translate the following natural language query to SQL:\n\n{text}\n\n### SQL Query:\n"
        inputs = tokenizer(
            input_prompt,
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True,
            padding=True
        )
        inputs = inputs.to(device)
        
        logger.info(f"Input prompt: {input_prompt}")
        logger.info(f"Tokenized input: {inputs['input_ids']}")
        
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=3,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated text: {generated_text}")
        
        sql_start = generated_text.find("### SQL Query:\n")
        if sql_start != -1:
            sql_text = generated_text[sql_start + len("### SQL Query:\n"):].strip()
            logger.info(f"Extracted SQL: {sql_text}")
            return sql_text
        else:
            logger.warning("SQL Query section not found in generated text")
            return generated_text.strip()
    except Exception as e:
        logger.error(f"Error generating SQL: {str(e)}")
        return f"Error generating SQL: {str(e)}"

# Initialize Flask and Swagger
app = Flask(__name__)
CORS(app)  # Enable CORS
swagger = Swagger(app, template={
    "swagger": "2.0",
    "info": {
        "title": "CodeS 1B SQL Generator API",
        "description": "API for generating SQL queries using CodeS 1B model",
        "version": "1.0.0"
    },
    "host": "localhost:5000",
    "basePath": "/",
    "schemes": ["http"],
})

# Global variables for model and tokenizer
model = None
tokenizer = None

@app.route("/generate_sql", methods=["POST"])
@swag_from({
    'tags': ['SQL Generation'],
    'description': 'Generate SQL query from natural language input',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Natural language query to convert to SQL',
                        'example': 'Give me all organizations'
                    }
                },
                'required': ['query'],
                'example': {
                    'query': 'Give me all organizations'
                }
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'SQL query generated successfully',
            'schema': {
                'type': 'object',
                'properties': {
                    'status': {'type': 'string'},
                    'query': {'type': 'string'},
                    'sql': {'type': 'string'}
                }
            }
        },
        '400': {
            'description': 'Invalid input or model not found'
        },
        '500': {
            'description': 'Server error'
        }
    }
})
def generate_sql_endpoint():
    logger.info("Received request to /generate_sql")
    global model, tokenizer

    if not os.path.exists(output_dir):
        return jsonify({"status": "error", "message": f"Model in directory {output_dir} not found. Train the model first."}), 400
    
    try:
        if model is None or tokenizer is None:
            logger.info(f"Loading model from {output_dir}")
            tokenizer = AutoTokenizer.from_pretrained(output_dir)
            model = AutoModelForCausalLM.from_pretrained(output_dir)
            model.to(device)
            # Set pad_token if not defined
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id
            logger.info("Model and tokenizer loaded")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return jsonify({"status": "error", "message": f"Error loading model: {str(e)}"}), 500

    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"status": "error", "message": "Parameter 'query' is missing"}), 400
    
    query = data["query"]
    try:
        generated_sql = generate_sql(query, model, tokenizer, max_input_length, max_new_tokens, device)
        return jsonify({"status": "success", "query": query, "sql": generated_sql}), 200
    except Exception as e:
        logger.error(f"Error generating SQL: {e}")
        return jsonify({"status": "error", "message": f"Error generating SQL: {str(e)}"}), 500
    

@app.route("/training_status", methods=["GET"])
@swag_from({
    'tags': ['Training Status'],
    'description': 'Get the current training status',
    'responses': {
        '200': {
            'description': 'Training status retrieved successfully',
            'schema': {
                'type': 'object',
                'properties': {
                    'status': {'type': 'string'},
                    'message': {'type': 'string'},
                    'training_status': {'type': 'object' }
                }
            }
        },
        '500': {
            'description': 'Server error'
        }
    }
})
def training_status_endpoint():
    logger.info("Received request to /training_status")
    try:
        status_file = "training_status.json"
        if os.path.exists(status_file):
            with open(status_file, "r") as f:
                status = json.load(f)
            return jsonify({
                "status": "success",
                "message": "Training status retrieved successfully",
                "training_status": status.get("training", {})
            }), 200
        else:
            return jsonify({
                "status": "success",
                "message": "No training status available",
                "training_status": {}
            }), 200
    except Exception as e:
        logger.error(f"Error retrieving training status: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error retrieving training status: {str(e)}"
        }), 500

@app.route("/training_logs", methods=["GET"])
@swag_from({
    'tags': ['Training Logs'],
    'description': 'Get the latest training logs',
    'parameters': [
        {
            'name': 'lines',
            'in': 'query',
            'type': 'integer',
            'required': False,
            'description': 'Number of log lines to retrieve (default: 10)',
            'default': 10
        }
    ],
    'responses': {
        '200': {
            'description': 'Training logs retrieved successfully',
            'schema': {
                'type': 'object',
                'properties': {
                    'status': {'type': 'string'},
                    'message': {'type': 'string'},
                    'logs': {'type': 'array', 'items': {'type': 'string'}}
                }
            }
        },
        '500': {
            'description': 'Server error'
        }
    }
})
def training_logs_endpoint():
    logger.info("Received request to /training_logs")
    try:
        lines = request.args.get('lines', default=10, type=int)
        log_file_path = "train_codes_1b_log.txt"
        logs = []
        
        if os.path.exists(log_file_path):
            with open(log_file_path, "r") as f:
                logs = f.readlines()
                logs = [line.strip() for line in logs[-lines:]]  # Get last N lines
            return jsonify({
                "status": "success",
                "message": "Logs retrieved successfully",
                "logs": logs
            }), 200
        else:
            return jsonify({
                "status": "success",
                "message": "No logs available",
                "logs": []
            }), 200
    except Exception as e:
        logger.error(f"Error retrieving logs: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error retrieving logs: {str(e)}"
        }), 500

@app.route("/train", methods=["POST"])
@swag_from({
    'tags': ['Model Training'],
    'description': 'Start training the model if it does not exist, or return a message if it already exists',
    'responses': {
        '200': {
            'description': 'Training started or model already exists',
            'schema': {
                'type': 'object',
                'properties': {
                    'status': {'type': 'string'},
                    'message': {'type': 'string'}
                }
            }
        },
        '500': {
            'description': 'Server error'
        }
    }
})
def train_endpoint():
    logger.info("Received request to /train")
    logger.debug(f"Request headers: {request.headers}")
    logger.debug(f"Request body: {request.get_data(as_text=True)}")
    try:
        if not os.path.exists(output_dir):
            logger.info(f"Model directory {output_dir} not found. Starting training...")
            training_result = start_training()
            logger.info(f"Training initiation result: {training_result}")
            return jsonify({
                "status": training_result["status"],
                "message": training_result["message"]
            }), 200
        else:
            logger.info(f"Model already exists in {output_dir}. No training needed.")
            return jsonify({
                "status": "success",
                "message": f"Model already exists in {output_dir}. No training needed."
            }), 200
    except Exception as e:
        logger.error(f"Error in training endpoint: {str(e)}")
        return jsonify({"status": "error", "message": f"Error in training endpoint: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)