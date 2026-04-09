import os
import openai
from openai import Stream, ChatCompletion

from .utils import print_debug

GEMINI_BASE_URL = os.environ.get(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)
DEFAULT_MODEL = os.environ.get("TYPEFLY_DEFAULT_MODEL", "gpt-4o")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

# Keep legacy aliases for compatibility with existing callers/UI toggles.
GPT3 = GEMINI_MODEL
GPT4 = GEMINI_MODEL
LLAMA3 = GEMINI_MODEL

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
chat_log_path = os.path.join(CURRENT_DIR, "assets/chat_log.txt")

def _mask_key(secret: str | None) -> str:
    value = "" if secret is None else str(secret).strip()
    if not value:
        return "(empty)"
    if len(value) <= 10:
        return f"{value[:2]}***{value[-2:]}"
    return f"{value[:6]}...{value[-4:]}"

class LLMWrapper:
    def __init__(self, temperature=0.0):
        self.temperature = temperature
        self.api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        ).strip()
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=GEMINI_BASE_URL,
        )
        print_debug(
            "[LLM-CONFIG] "
            f"has_gemini_key={bool(os.environ.get('GEMINI_API_KEY'))} "
            f"has_google_key={bool(os.environ.get('GOOGLE_API_KEY'))} "
            f"selected_key_len={len(self.api_key)} "
            f"selected_key_masked={_mask_key(self.api_key)} "
            f"base_url={GEMINI_BASE_URL}"
        )

    def request(self, prompt, model_name=GEMINI_MODEL, stream=False) -> str | Stream[ChatCompletion.ChatCompletionChunk]:
        selected_model = str(model_name or GEMINI_MODEL)
        print_debug(
            "[LLM-REQUEST] "
            f"model={selected_model} "
            f"base_url={GEMINI_BASE_URL} "
            f"key_len={len(self.api_key)} "
            f"key_masked={_mask_key(self.api_key)}"
        )

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
