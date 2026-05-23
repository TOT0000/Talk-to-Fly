import os
import openai
from openai import Stream, ChatCompletion

from .utils import print_debug

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")

OPENAI_DEFAULT_MODEL = os.environ.get("OPENAI_DEFAULT_MODEL", "gpt-4o")
GEMINI_DEFAULT_MODEL = os.environ.get("GEMINI_DEFAULT_MODEL", "gemini-2.5-flash")
LMSTUDIO_DEFAULT_MODEL = os.environ.get("LMSTUDIO_DEFAULT_MODEL", "local-model")


def _normalize_provider(raw: str | None) -> str:
    value = str(raw or "").strip().lower()
    if value in {"openai", "gemini", "lmstudio"}:
        return value
    return ""


def _resolve_default_provider() -> str:
    forced = _normalize_provider(os.environ.get("LLM_PROVIDER"))
    if forced:
        return forced
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    return "gemini"


def _provider_default_model(provider: str) -> str:
    if provider == "openai":
        return OPENAI_DEFAULT_MODEL
    if provider == "lmstudio":
        return LMSTUDIO_DEFAULT_MODEL
    return GEMINI_DEFAULT_MODEL


def resolve_runtime_provider_config() -> dict:
    provider = _resolve_default_provider()
    openai_base_url = os.environ.get("OPENAI_BASE_URL", OPENAI_BASE_URL)
    gemini_base_url = os.environ.get("GEMINI_BASE_URL", GEMINI_BASE_URL)
    lmstudio_base_url = os.environ.get("LMSTUDIO_BASE_URL", LMSTUDIO_BASE_URL)
    base_url = (
        openai_base_url
        if provider == "openai"
        else (lmstudio_base_url if provider == "lmstudio" else gemini_base_url)
    )
    return {
        "provider": provider,
        "base_url": str(base_url or "").strip(),
        "lmstudio_base_url": str(lmstudio_base_url or "").strip(),
    }


DEFAULT_PROVIDER = _resolve_default_provider()
GEMINI_MODEL = _provider_default_model(DEFAULT_PROVIDER)
MODEL_NAME = GEMINI_MODEL

# Keep legacy aliases for compatibility with existing callers/UI toggles.
GPT3 = MODEL_NAME
GPT4 = MODEL_NAME
LLAMA3 = MODEL_NAME

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
        runtime = resolve_runtime_provider_config()
        self.provider = runtime["provider"]
        self.base_url = runtime["base_url"]
        self._lmstudio_base_url = runtime["lmstudio_base_url"]
        self._enforce_lmstudio = str(os.environ.get("TYPEFLY_ENFORCE_LMSTUDIO", "")).strip().lower() in {"1", "true", "yes"}
        if self._enforce_lmstudio and self.provider != "lmstudio":
            raise RuntimeError(f"provider_not_lmstudio: provider={self.provider}")
        if self._enforce_lmstudio and (not self.base_url):
            raise RuntimeError("provider_not_lmstudio: empty_base_url")
        if self._enforce_lmstudio and (self.base_url != self._lmstudio_base_url):
            raise RuntimeError(
                f"provider_not_lmstudio: base_url_mismatch base_url={self.base_url} lmstudio_base_url={self._lmstudio_base_url}"
            )
        if self.provider == "openai":
            self.api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
            self.key_source = ("OPENAI_API_KEY" if self.api_key else "(empty)")
        elif self.provider == "lmstudio":
            self.api_key = (os.environ.get("LMSTUDIO_API_KEY") or "lmstudio").strip()
            self.key_source = "LMSTUDIO_API_KEY"
        else:
            self.api_key = (
                os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or ""
            ).strip()
            if os.environ.get("GEMINI_API_KEY"):
                self.key_source = "GEMINI_API_KEY"
            elif os.environ.get("GOOGLE_API_KEY"):
                self.key_source = "GOOGLE_API_KEY"
            else:
                self.key_source = "(empty)"
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        print_debug(
            "[LLM-CONFIG] "
            f"provider={self.provider} "
            f"has_gemini_key={bool(os.environ.get('GEMINI_API_KEY'))} "
            f"has_openai_key={bool(os.environ.get('OPENAI_API_KEY'))} "
            f"has_google_key={bool(os.environ.get('GOOGLE_API_KEY'))} "
            f"selected_key_len={len(self.api_key)} "
            f"selected_key_masked={_mask_key(self.api_key)} "
            f"key_source={self.key_source} "
            f"base_url={self.base_url} "
            f"default_model={_provider_default_model(self.provider)}"
        )

    def _resolve_request_route(self, model_name) -> dict:
        raw = "" if model_name is None else str(model_name).strip()
        if raw == "":
            return {
                "provider": "openai",
                "model": os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o"),
                "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                "api_key": os.getenv("OPENAI_API_KEY", "").strip(),
                "key_source": "OPENAI_API_KEY",
                "model_source": "empty_ui_default_openai",
            }
        return {
            "provider": "lmstudio",
            "model": raw,
            "base_url": os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
            "api_key": os.getenv("LMSTUDIO_API_KEY", "lmstudio").strip(),
            "key_source": "LMSTUDIO_API_KEY",
            "model_source": "ui_input_lmstudio",
        }

    def request(self, prompt, model_name=GEMINI_MODEL, stream=False) -> str | Stream[ChatCompletion.ChatCompletionChunk]:
        route = self._resolve_request_route(model_name)
        provider = str(route["provider"])
        selected_model = str(route["model"])
        base_url = str(route["base_url"])
        api_key = str(route["api_key"])
        key_source = str(route["key_source"])
        model_source = str(route["model_source"])
        if provider == "openai" and not api_key:
            raise RuntimeError("missing_openai_api_key_for_default_gpt4o")
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        print_debug(f"[LLM] route_provider={provider}")
        print_debug(f"[LLM] route_model={selected_model}")
        print_debug(f"[LLM] route_base_url={base_url}")
        print_debug(f"[LLM] route_key_source={key_source}")
        print_debug(f"[LLM] model_source={model_source}")

        with open(chat_log_path, "a") as f:
            f.write(prompt + "\n---\n")
        print_debug(f"[LLM] Prompt written to {chat_log_path}")
        
        try:
            response = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                stream=stream,
            )
        except Exception as exc:
            if provider == "lmstudio":
                print_debug(f"[LLM-ERROR] provider=lmstudio base_url={base_url} model={selected_model} error={exc}")
            message = str(exc or "")
            if "Missing or invalid Authorization header." in message:
                raise RuntimeError("invalid_authorization_header") from exc
            raise

        # save the message in a txt
        with open(chat_log_path, "a") as f:
            if not stream:
                f.write(response.model_dump_json(indent=2) + "\n---\n")

        if stream:
            return response

        message = response.choices[0].message
        content = str(getattr(message, "content", "") or "")
        reasoning_content = None
        if hasattr(message, "reasoning_content"):
            reasoning_content = getattr(message, "reasoning_content")
        if (not reasoning_content) and hasattr(message, "model_extra"):
            extras = getattr(message, "model_extra") or {}
            if isinstance(extras, dict):
                reasoning_content = extras.get("reasoning_content")
        if (not content.strip()) and str(reasoning_content or "").strip():
            raise RuntimeError("reasoning_only_empty_content")
        return content
