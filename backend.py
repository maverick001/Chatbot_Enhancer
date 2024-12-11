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

# Add Together AI configuration
TOGETHER_API_KEY = "0930f917a67487bbf40710d622c82ea2dd998c71b9a5f443fbd8cb418505cc34"
TOGETHER_API_BASE = "https://api.together.xyz/inference"

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
        
        system_prompt = """Please provide very concise responses strictly limited to 150 tokens. 
        Keep your answers clear and brief. Consider the conversation history provided when responding, 
        but ensure your response does not exceed 150 tokens under any circumstances."""
        
        complete_prompt = f"{system_prompt}\n\nConversation history:\n{user_prompt}"
        
        app.logger.info(f"Received full prompt with history")
        app.logger.info(f"Using models: {model1} and {model2}")
        
        def generate():
            responses = {"response1": "", "response2": ""}
            tokens = {"response1": 0, "response2": 0}
            
            # Get response from first model
            app.logger.info(f"Starting call to {model1} model...")
            try:
                response1 = requests.post(
                    OLLAMA_API_BASE, 
                    json={
                        "model": model1,
                        "prompt": complete_prompt,
                        "stream": True,
                        "options": {
                            "num_tokens": 150,
                            "stop": ["</s>", "<|im_end|>"],  # Add stop tokens
                            "temperature": 0.5
                        }
                    },
                    stream=True,
                    timeout=30
                )
                
                for line in response1.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line)
                            response_text = json_response.get('response', '')
                            if response_text:
                                responses["response1"] += response_text
                                tokens["response1"] += 1
                                yield f"data: {json.dumps({'response1': response_text, 'tokens1': tokens['response1']})}\n\n"
                        except json.JSONDecodeError:
                            continue

            except requests.exceptions.RequestException as e:
                app.logger.error(f"Error with {model1}: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

            # Get response from second model
            app.logger.info(f"Starting call to {model2} model...")
            try:
                response2 = requests.post(
                    OLLAMA_API_BASE, 
                    json={
                        "model": model2,
                        "prompt": complete_prompt,
                        "stream": True,
                        "options": {
                            "num_tokens": 150,
                            "stop": ["</s>", "<|im_end|>"],  # Add stop tokens
                            "temperature": 0.5
                        }
                    },
                    stream=True,
                    timeout=30
                )
                
                for line in response2.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line)
                            response_text = json_response.get('response', '')
                            if response_text:
                                responses["response2"] += response_text
                                tokens["response2"] += 1
                                yield f"data: {json.dumps({'response2': response_text, 'tokens2': tokens['response2']})}\n\n"
                        except json.JSONDecodeError:
                            continue

            except requests.exceptions.RequestException as e:
                app.logger.error(f"Error with {model2}: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

            app.logger.info("Completed streaming both responses")
            yield f"data: {json.dumps({'done': True, 'final_tokens': tokens})}\n\n"

        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/get_summary", methods=["POST"])
def get_summary():
    try:
        data = request.json
        if not data or 'prompt' not in data:
            return jsonify({"error": "No prompt provided"}), 400

        def generate():
            try:
                # Enhanced prompt for more natural output
                formatted_prompt = f"""<|im_start|>system
You are a helpful AI assistant that excels at comparing and synthesizing information. 
Your task is to provide clear, well-written summaries in natural language.
Avoid using bullet points or numbered lists.
Write in a flowing, coherent paragraph style.
<|im_end|>

<|im_start|>user
{data['prompt']}
Please provide your analysis in a natural, conversational style, focusing on the key common points.
<|im_end|>

<|im_start|>assistant
Based on comparing these responses, here's a coherent synthesis:"""
                
                response = requests.post(
                    TOGETHER_API_BASE,
                    headers={
                        "Authorization": f"Bearer {TOGETHER_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                        "prompt": formatted_prompt,
                        "max_tokens": 150,
                        "temperature": 0.5,
                        "stream": True,
                        "top_p": 0.7,
                        "repetition_penalty": 1.1,
                        "stop": ["<|im_end|>"]
                    },
                    stream=True
                )

                if response.status_code != 200:
                    error_msg = f"Together AI API error: {response.status_code}"
                    app.logger.error(error_msg)
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
                    return

                app.logger.info("Starting to process Together AI response stream")
                
                buffer = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            # Decode bytes to string
                            line_text = line.decode('utf-8') if isinstance(line, bytes) else line
                            app.logger.debug(f"Received line: {line_text}")

                            # Handle different response formats
                            try:
                                json_response = json.loads(line_text)
                                if isinstance(json_response, dict):
                                    if 'text' in json_response:
                                        text = json_response['text']
                                    elif 'token' in json_response:
                                        text = json_response['token'].get('text', '')
                                    elif 'output' in json_response:
                                        text = json_response['output']
                                    else:
                                        text = ''
                                    
                                    if text:
                                        # Remove any remaining special tokens
                                        text = text.replace("<|im_end|>", "").strip()
                                        if text:
                                            buffer += text
                                            yield f"data: {json.dumps({'summary': text})}\n\n"
                            except json.JSONDecodeError:
                                # If not JSON, try to use the line directly
                                if line_text.strip():
                                    clean_text = line_text.replace("<|im_end|>", "").strip()
                                    if clean_text:
                                        yield f"data: {json.dumps({'summary': clean_text})}\n\n"

                        except Exception as e:
                            app.logger.error(f"Error processing line: {str(e)}")
                            app.logger.error(f"Problematic line: {line}")
                            continue

                app.logger.info("Completed processing Together AI response stream")
                if buffer:
                    app.logger.info(f"Final summary buffer: {buffer}")

            except requests.exceptions.RequestException as e:
                error_msg = f"Error calling Together AI: {str(e)}"
                app.logger.error(error_msg)
                yield f"data: {json.dumps({'error': error_msg})}\n\n"

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
