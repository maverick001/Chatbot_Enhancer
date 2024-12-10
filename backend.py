## Backend script for the 2-window chatbot

from flask import Flask, request, jsonify, send_file, Response
import requests
import logging
import os
import json
import subprocess

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)  # Enable detailed logging

# Remove the model loading code and replace with Ollama API endpoints
# Update the API endpoint
OLLAMA_API_BASE = "http://localhost:11434/api/generate"

@app.route("/")
def home():
    return send_file('index.html')  # Make sure index.html is in the same directory as backend.py

@app.route("/get_models")
def get_models():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            # Parse the output to get model names
            lines = result.stdout.strip().split('\n')[1:]  # Skip header line
            models = [line.split()[0] for line in lines if line]
            return jsonify(models)
    except Exception as e:
        app.logger.error(f"Error getting models: {str(e)}")
        return jsonify([])

@app.route("/get_response", methods=["POST"])
def get_response():
    try:
        data = request.json
        if not data or 'prompt' not in data:
            app.logger.error("No prompt provided in request")
            return jsonify({"error": "No prompt provided"}), 400
        
        user_prompt = data.get("prompt")
        model1 = data.get("model1", "llama3.2")
        model2 = data.get("model2", "mistral")
        
        # Add system prompt with token limit
        system_prompt = "Please provide concise responses limited to approximately 150 tokens. Keep your answers clear but brief."
        full_prompt = f"{system_prompt}\n\nUser: {user_prompt}"
        
        app.logger.info(f"Received prompt: {user_prompt}")
        app.logger.info(f"Using models: {model1} and {model2}")
        
        def generate():
            responses = {"response1": "", "response2": ""}
            
            # Get response from first model
            app.logger.info(f"Starting call to {model1} model...")
            try:
                response1 = requests.post(
                    OLLAMA_API_BASE, 
                    json={
                        "model": model1,
                        "prompt": full_prompt,  # Use the prompt with system instruction
                        "stream": True,
                        "options": {
                            "num_tokens": 150  # Add token limit parameter
                        }
                    },
                    stream=True,
                    timeout=30
                )
                app.logger.info(f"{model1} API call successful")
                
                for line in response1.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line)
                            chunk = json_response.get('response', '')
                            responses["response1"] += chunk
                            app.logger.debug(f"Streaming chunk from {model1}")
                            yield f"data: {json.dumps(responses)}\n\n"
                        except json.JSONDecodeError as e:
                            app.logger.error(f"JSON decode error for {model1}: {str(e)}")
                            
            except requests.exceptions.RequestException as e:
                app.logger.error(f"Error calling {model1}: {str(e)}")
                yield f"data: {json.dumps({'error': f'Error with {model1}: {str(e)}'})}\n\n"

            # Get response from second model
            app.logger.info(f"Starting call to {model2} model...")
            try:
                response2 = requests.post(
                    OLLAMA_API_BASE, 
                    json={
                        "model": model2,
                        "prompt": full_prompt,  # Use the prompt with system instruction
                        "stream": True,
                        "options": {
                            "num_tokens": 150  # Add token limit parameter
                        }
                    },
                    stream=True,
                    timeout=30
                )
                app.logger.info(f"{model2} API call successful")
                
                for line in response2.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line)
                            chunk = json_response.get('response', '')
                            responses["response2"] += chunk
                            app.logger.debug(f"Streaming chunk from {model2}")
                            yield f"data: {json.dumps(responses)}\n\n"
                        except json.JSONDecodeError as e:
                            app.logger.error(f"JSON decode error for {model2}: {str(e)}")
                            
            except requests.exceptions.RequestException as e:
                app.logger.error(f"Error calling {model2}: {str(e)}")
                yield f"data: {json.dumps({'error': f'Error with {model2}: {str(e)}'})}\n\n"

            app.logger.info("Completed streaming both responses")

        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
