## Backend script for the 2-window chatbot

from flask import Flask, request, jsonify, send_file, Response
import requests
import logging
import os
import json

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)  # Enable detailed logging

# Remove the model loading code and replace with Ollama API endpoints
# Update the API endpoint
OLLAMA_API_BASE = "http://localhost:11434/api/generate"

@app.route("/")
def home():
    return send_file('index.html')  # Make sure index.html is in the same directory as backend.py

@app.route("/get_response", methods=["POST"])
def get_response():
    try:
        data = request.json
        if not data or 'prompt' not in data:
            app.logger.error("No prompt provided in request")
            return jsonify({"error": "No prompt provided"}), 400
        
        prompt = data.get("prompt")
        app.logger.info(f"Received prompt: {prompt}")

        def generate():
            # Get response from llama3.2
            app.logger.info("Calling llama3.2 model...")
            try:
                response1 = requests.post(
                    OLLAMA_API_BASE, 
                    json={
                        "model": "llama3.2",
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
                app.logger.error(f"Error calling llama3.2: {str(e)}")
                yield f"data: {json.dumps({'error': f'Error with llama3.2: {str(e)}'})}\n\n"
                return

            # Get response from mistral
            app.logger.info("Calling mistral model...")
            try:
                response2 = requests.post(
                    OLLAMA_API_BASE, 
                    json={
                        "model": "mistral",
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
                app.logger.error(f"Error calling mistral: {str(e)})")
                yield f"data: {json.dumps({'error': f'Error with mistral: {str(e)}'})}\n\n"
                return

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
