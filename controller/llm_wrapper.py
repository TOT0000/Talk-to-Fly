import os
import openai
from openai import Stream, ChatCompletion

from .utils import print_debug

GEMINI_BASE_URL = os.environ.get(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Keep legacy aliases for compatibility with existing callers/UI toggles.
GPT3 = GEMINI_MODEL
GPT4 = GEMINI_MODEL
LLAMA3 = GEMINI_MODEL

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
chat_log_path = os.path.join(CURRENT_DIR, "assets/chat_log.txt")

class LLMWrapper:
    def __init__(self, temperature=0.0):
        self.temperature = temperature
        self.client = openai.OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url=GEMINI_BASE_URL,
        )

    def request(self, prompt, model_name=GEMINI_MODEL, stream=False) -> str | Stream[ChatCompletion.ChatCompletionChunk]:
        selected_model = str(model_name or GEMINI_MODEL)

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
