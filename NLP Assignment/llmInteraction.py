import os
import requests
import json
import datetime

from dotenv import load_dotenv

load_dotenv()

# Load the API key from the environment variable
if os.getenv("OPENROUTER_API_KEY") is None or os.getenv("OPENROUTER_API_KEY") == "":
    print("OPENROUTER_API_KEY is not set")
    exit(1)
else:
    api_key = os.getenv("OPENROUTER_API_KEY")

import time
import json
import requests

def load_llm_model(query, model, retries=3, retry_delay=2):
    def stream_response():
        url = "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "assistant", "content": "You are a helpful assistant. You will answer based on the user's request."},
                {"role": "user", "content": query}
            ],
            "stream": True
        }

        for attempt in range(retries):
            try:
                buffer = ""
                with requests.post(url, headers=headers, json=payload, stream=True, timeout=30) as r:
                    r.encoding = 'utf-8'
                    for chunk in r.iter_content(chunk_size=1024, decode_unicode=True):
                        buffer += chunk
                        while True:
                            line_end = buffer.find('\n')
                            if line_end == -1:
                                break
                            line = buffer[:line_end].strip()
                            buffer = buffer[line_end + 1:]

                            if line.startswith('data: '):
                                data = line[6:]
                                if data == '[DONE]':
                                    return
                                try:
                                    data_obj = json.loads(data)

                                    # Handle API errors
                                    if 'error' in data_obj:
                                        error = data_obj['error']
                                        message = error.get('message', '')
                                        print("API Error:", message)

                                        if 'Rate limit exceeded' in message:
                                            reset_ms = error.get('metadata', {}).get('headers', {}).get('X-RateLimit-Reset')
                                            if reset_ms:
                                                reset_timestamp = int(reset_ms) // 1000
                                                reset_time = datetime.datetime.fromtimestamp(reset_timestamp)
                                                yield f"⚠️ Rate limit exceeded. Please wait until {reset_time.strftime('%Y-%m-%d %H:%M:%S')} to try again."
                                            else:
                                                yield "⚠️ Rate limit exceeded. Please try again later."
                                            return  # STOP the generator
                                        else:
                                            yield f"❌ API error: {message}"
                                            return

                                    choices = data_obj.get("choices", [])
                                    if choices:
                                        content = choices[0].get("delta", {}).get("content")
                                        if content:
                                            yield content
                                    else:
                                        print("Unexpected data format:", data_obj)
                                        yield "⚠️ Unexpected response format."
                                        return
                                except (json.JSONDecodeError, KeyError) as e:
                                    print(f"⚠️ Failed to decode chunk: {e}")
                                    continue
                break  # success
            except requests.RequestException as e:
                print(f"Network error: {e}")
                if attempt < retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    yield f"❌ API request failed after {retries} attempts. Error: {str(e)}"
                    return
    return stream_response