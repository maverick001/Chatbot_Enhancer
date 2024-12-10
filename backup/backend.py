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
            return jsonify({"error": "No prompt provided"}), 400
        
        prompt = data.get("prompt")
        model1 = data.get("model1", "llama3.2")  # Default fallback
        model2 = data.get("model2", "mistral")   # Default fallback
        
        app.logger.info(f"Received prompt: {prompt}")
        
        def generate():
            # Get response from first model
            app.logger.info(f"Calling {model1} model...")
            try:
                response1 = requests.post(
                    OLLAMA_API_BASE, 
                    json={
                        "model": model1,
                        "prompt": prompt,
                        "stream": True
                    },
                    stream=True
                )
                response1_text = ""
                for line in response1.iter_lines():
                    if line:
                        json_response = json.loads(line)
                        response1_text += json_response.get('response', '')
                        yield f"data: {json.dumps({'response1': response1_text})}\n\n"

            except requests.exceptions.RequestException as e:
                app.logger.error(f"Error calling {model1}: {str(e)}")
                yield f"data: {json.dumps({'error': f'Error with {model1}: {str(e)}'})}\n\n"
                return

            # Get response from second model
            app.logger.info(f"Calling {model2} model...")
            try:
                response2 = requests.post(
                    OLLAMA_API_BASE, 
                    json={
                        "model": model2,
                        "prompt": prompt,
                        "stream": True
                    },
                    stream=True
                )
                response2_text = ""
                for line in response2.iter_lines():
                    if line:
                        json_response = json.loads(line)
                        response2_text += json_response.get('response', '')
                        yield f"data: {json.dumps({'response2': response2_text})}\n\n"

            except requests.exceptions.RequestException as e:
                app.logger.error(f"Error calling {model2}: {str(e)}")
                yield f"data: {json.dumps({'error': f'Error with {model2}: {str(e)}'})}\n\n"
                return

        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
