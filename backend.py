from flask import Flask, request, jsonify, Response, send_from_directory
import os
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict, List
import asyncio
from queue import Queue
import threading
import requests
import json

app = Flask(__name__, static_folder='.')
OLLAMA_API_BASE = "http://localhost:11434/api/generate"

# Serve index.html at root
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# Serve any other static files
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

# Custom streaming callback handler
class StreamingHandler(BaseCallbackHandler):
    def __init__(self, queue: Queue):
        self.queue = queue
        self.streaming_text = ""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.streaming_text += token
        self.queue.put(token)

# Define response schemas
response_schemas = [
    ResponseSchema(
        name="common_points",
        description="Key points that are present in both responses"
    ),
    ResponseSchema(
        name="synthesized_summary",
        description="A well-written paragraph synthesizing the common points"
    )
]

# Create output parser
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Create comparison prompt template
COMPARE_TEMPLATE = """
You are a precise and analytical AI assistant. Your task is to compare two AI responses and synthesize their common ground.

Question asked: {question}

Response 1:
{response1}

Response 2:
{response2}

Instructions:
1. Identify the key points that are present in both responses
2. Synthesize these common points into a coherent summary
3. Ignore any contradicting or unique points
4. Keep the response concise and focused

{format_instructions}

Provide your analysis:
"""

# Initialize prompt template
prompt = PromptTemplate(
    template=COMPARE_TEMPLATE,
    input_variables=["question", "response1", "response2"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)

@app.route("/get_summary", methods=["POST"])
def get_summary():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        question = data.get("question", "")
        response1 = data.get("response1", "")
        response2 = data.get("response2", "")

        # Initialize queue for streaming
        queue = Queue()
        streaming_handler = StreamingHandler(queue)

        # Initialize LangChain components
        llm = Ollama(
            model="mixtral",
            callbacks=[streaming_handler],
            temperature=0.5,
            streaming=True
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)

        def generate():
            try:
                # Run the chain in a separate thread
                def run_chain():
                    chain.run(
                        question=question,
                        response1=response1,
                        response2=response2
                    )

                thread = threading.Thread(target=run_chain)
                thread.start()

                # Stream the results
                while thread.is_alive() or not queue.empty():
                    try:
                        token = queue.get(timeout=0.1)
                        yield f"data: {json.dumps({'summary': token})}\n\n"
                    except:
                        continue

            except Exception as e:
                app.logger.error(f"Error in generate: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Add this new route to get available models
@app.route('/get_models', methods=['GET'])
def get_models():
    try:
        # Call Ollama API to get available models
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            data = response.json()
            # Extract model names from the response
            if 'models' in data and isinstance(data['models'], list):
                model_names = [model['name'] for model in data['models']]
            else:
                # Fallback for different API response format
                model_names = list(data.keys()) if isinstance(data, dict) else []
            
            app.logger.info(f"Available models: {model_names}")
            return jsonify({"models": model_names})
        else:
            app.logger.error(f"Failed to fetch models: {response.status_code}")
            return jsonify({"error": "Failed to fetch models from Ollama"}), 500
    except Exception as e:
        app.logger.error(f"Error fetching models: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Add favicon route to handle the 404
@app.route('/favicon.ico')
def favicon():
    return '', 204  # Return no content

@app.route("/get_response", methods=["POST"])
def get_response():
    try:
        data = request.json
        TOKEN_LIMIT = 200

        # Add system prompt for concise responses
        SYSTEM_PROMPT = """Please provide clear and concise responses within 200 tokens. 
        Focus on the most important information and be direct in your answers. 
        If a longer response is needed, prioritize the most crucial points first.
        In no chance shall an answer be longer than 250 tokens."""

        # Combine system prompt with user prompt
        combined_prompt = f"System: {SYSTEM_PROMPT}\n\nUser: {data['prompt']}"
        
        def generate():
            try:
                responses = {"response1": "", "response2": ""}
                tokens = {"response1": 0, "response2": 0}

                # Model 1 response with system prompt
                response1 = requests.post(
                    OLLAMA_API_BASE,
                    json={
                        "model": data['model1'],
                        "prompt": combined_prompt,
                        "system": SYSTEM_PROMPT,  # Add system prompt to Ollama context
                        "stream": True
                    },
                    stream=True
                )

                for line in response1.iter_lines():
                    if line and tokens["response1"] < TOKEN_LIMIT:
                        try:
                            json_response = json.loads(line)
                            response_text = json_response.get('response', '')
                            if response_text:
                                responses["response1"] += response_text
                                tokens["response1"] += 1
                                yield f"data: {json.dumps({'response1': response_text, 'tokens1': tokens['response1']})}\n\n"
                                
                                if tokens["response1"] >= TOKEN_LIMIT:
                                    message = json.dumps({'response1': '\n[Token limit reached: 200]', 'tokens1': tokens['response1']})
                                    yield f"data: {message}\n\n"
                                    break
                        except json.JSONDecodeError:
                            continue

                # Model 2 response with system prompt
                response2 = requests.post(
                    OLLAMA_API_BASE,
                    json={
                        "model": data['model2'],
                        "prompt": combined_prompt,
                        "system": SYSTEM_PROMPT,  # Add system prompt to Ollama context
                        "stream": True
                    },
                    stream=True
                )

                for line in response2.iter_lines():
                    if line and tokens["response2"] < TOKEN_LIMIT:
                        try:
                            json_response = json.loads(line)
                            response_text = json_response.get('response', '')
                            if response_text:
                                responses["response2"] += response_text
                                tokens["response2"] += 1
                                yield f"data: {json.dumps({'response2': response_text, 'tokens2': tokens['response2']})}\n\n"
                                
                                if tokens["response2"] >= TOKEN_LIMIT:
                                    message = json.dumps({'response2': '\n[Token limit reached: 200]', 'tokens2': tokens['response2']})
                                    yield f"data: {message}\n\n"
                                    break
                        except json.JSONDecodeError:
                            continue

                yield f"data: {json.dumps({'done': True, 'final_tokens': tokens})}\n\n"

            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                app.logger.error(error_msg)
                yield f"data: {json.dumps({'error': error_msg})}\n\n"

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Keep existing routes and functions
# ... rest of your code ...

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)