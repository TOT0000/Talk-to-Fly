import os
import openai
from openai import Stream, ChatCompletion

from .utils import print_debug

LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.environ.get("LM_STUDIO_MODEL", "google/gemma-3-4b")

# Keep legacy aliases for compatibility with existing callers/UI toggles.
GPT3 = LM_STUDIO_MODEL
GPT4 = LM_STUDIO_MODEL
LLAMA3 = "meta-llama/Meta-Llama-3-8B-Instruct"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
chat_log_path = os.path.join(CURRENT_DIR, "assets/chat_log.txt")

class LLMWrapper:
    def __init__(self, temperature=0.0):
        self.temperature = temperature
        self.client = openai.OpenAI(
            base_url=LM_STUDIO_BASE_URL,
            api_key="lm-studio",
        )

    def request(self, prompt, model_name=LM_STUDIO_MODEL, stream=False) -> str | Stream[ChatCompletion.ChatCompletionChunk]:
        selected_model = str(model_name or LM_STUDIO_MODEL)

        with open(chat_log_path, "a") as f:
            f.write(prompt + "\n---\n")
        print_debug(f"[LLM] Prompt written to {chat_log_path}")
        
        response = self.client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stream=stream,
        )

        # save the message in a txt
        with open(chat_log_path, "a") as f:
            if not stream:
                f.write(response.model_dump_json(indent=2) + "\n---\n")

        if stream:
            return response

        return response.choices[0].message.content
