"""
llm_adapters.py — Unified LLM backend abstraction for HinduMind agents.

Adapters:
  gemini       — Google Gemini (free API, recommended)
  groq         — Groq API (free tier, ultra-fast — llama3 / mixtral)
  ollama       — Local Ollama server (free, local GPU/CPU)
  openai       — OpenAI GPT-4o etc.
  anthropic    — Anthropic Claude
  huggingface  — Local HuggingFace pipeline
  mock         — Offline canned responses for testing

All adapters expose: .generate(prompt, system_prompt) -> str
"""

import os
from abc import ABC, abstractmethod


class BaseLLMAdapter(ABC):
    """Abstract base for all LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a response given a user prompt and optional system prompt."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether this backend is reachable."""
        ...


# ─────────────────────────────────────────────────────────────────
# Ollama (local, free)
# ─────────────────────────────────────────────────────────────────

class OllamaAdapter(BaseLLMAdapter):
    """
    Calls a local Ollama server (http://localhost:11434).
    Install: https://ollama.com  —  then: ollama pull mistral
    """

    def __init__(self, model: str = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "mistral")
        self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

    def is_available(self) -> bool:
        try:
            import requests
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        import requests
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt}
            ],
            "stream": False
        }
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as e:
            return f"[OllamaAdapter ERROR] {e}"


# ─────────────────────────────────────────────────────────────────
# OpenAI
# ─────────────────────────────────────────────────────────────────

class OpenAIAdapter(BaseLLMAdapter):
    """Calls OpenAI chat completions API (GPT-4o / GPT-3.5 etc)."""

    def __init__(self, model: str = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.api_key = os.getenv("OPENAI_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            resp = client.chat.completions.create(model=self.model, messages=messages)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[OpenAIAdapter ERROR] {e}"


# ─────────────────────────────────────────────────────────────────
# HuggingFace (local pipeline)
# ─────────────────────────────────────────────────────────────────

class HuggingFaceAdapter(BaseLLMAdapter):
    """
    Runs a local HuggingFace text-generation pipeline.
    Default: mistralai/Mistral-7B-Instruct-v0.2 (requires ~14GB RAM).
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv(
            "HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"
        )
        self._pipeline = None

    def is_available(self) -> bool:
        try:
            import transformers  # noqa
            return True
        except ImportError:
            return False

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                max_new_tokens=512,
                device_map="auto"
            )

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        try:
            self._load()
            full_prompt = f"<<SYS>>{system_prompt}<</SYS>>\n\n{prompt}" if system_prompt else prompt
            out = self._pipeline(full_prompt)
            return out[0]["generated_text"].replace(full_prompt, "").strip()
        except Exception as e:
            return f"[HuggingFaceAdapter ERROR] {e}"


# ─────────────────────────────────────────────────────────────────
# Google Gemini  (free API — RECOMMENDED)
# Get key: https://aistudio.google.com/app/apikey
# ─────────────────────────────────────────────────────────────────

class GeminiAdapter(BaseLLMAdapter):
    """
    Google Gemini via google-generativeai SDK or direct REST.
    Free tier: 60 req/min on gemini-1.5-flash (as of 2025).

    Setup:
      pip install google-generativeai
      Set GEMINI_API_KEY in .env
    """

    def __init__(self, model: str = None):
        self.model   = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.api_key = os.getenv("GEMINI_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if not self.api_key:
            return "[GeminiAdapter ERROR] GEMINI_API_KEY not set"

        import requests, time
        url = (f"https://generativelanguage.googleapis.com/v1/models/"
               f"{self.model}:generateContent?key={self.api_key}")

        # Prepend system prompt to user content (works reliably on v1 REST)
        full_text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        payload = {"contents": [{"role": "user", "parts": [{"text": full_text}]}]}

        for attempt in range(3):
            try:
                r = requests.post(url, json=payload, timeout=60)
                if r.status_code == 429:
                    wait = 10 * (attempt + 1)
                    print(f"[GeminiAdapter] Rate limited — retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            except requests.exceptions.HTTPError as e:
                return f"[GeminiAdapter HTTP ERROR] {e} — {r.text[:200]}"
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                    continue
                return f"[GeminiAdapter ERROR] {e}"

        return "[GeminiAdapter ERROR] Rate limit — try again in 1 minute"


# ─────────────────────────────────────────────────────────────────
# Groq  (free tier, extremely fast)
# Get key: https://console.groq.com → API Keys
# ─────────────────────────────────────────────────────────────────

class GroqAdapter(BaseLLMAdapter):
    """
    Groq cloud inference — runs llama3, mixtral, gemma at very high speed.
    Free tier: 14,400 tokens/min on most models.

    Setup:
      pip install groq
      Set GROQ_API_KEY in .env
    """

    DEFAULT_MODELS = ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"]

    def __init__(self, model: str = None):
        self.model   = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.api_key = os.getenv("GROQ_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if not self.api_key:
            return "[GroqAdapter ERROR] GROQ_API_KEY not set"
        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            resp = client.chat.completions.create(model=self.model, messages=messages)
            return resp.choices[0].message.content.strip()
        except ImportError:
            # REST fallback
            pass
        except Exception as e:
            return f"[GroqAdapter SDK ERROR] {e}"

        try:
            import requests
            headers = {"Authorization": f"Bearer {self.api_key}",
                       "Content-Type": "application/json"}
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            payload = {"model": self.model, "messages": messages}
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                              json=payload, headers=headers, timeout=60)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[GroqAdapter REST ERROR] {e}"


# ─────────────────────────────────────────────────────────────────
# Anthropic Claude
# Get key: https://console.anthropic.com
# ─────────────────────────────────────────────────────────────────

class AnthropicAdapter(BaseLLMAdapter):
    """
    Anthropic Claude via REST API.
    Setup:
      pip install anthropic
      Set ANTHROPIC_API_KEY in .env
    """

    def __init__(self, model: str = None):
        self.model   = model or os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if not self.api_key:
            return "[AnthropicAdapter ERROR] ANTHROPIC_API_KEY not set"
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            kwargs = {"model": self.model, "max_tokens": 1024,
                      "messages": [{"role": "user", "content": prompt}]}
            if system_prompt:
                kwargs["system"] = system_prompt
            msg = client.messages.create(**kwargs)
            return msg.content[0].text.strip()
        except ImportError:
            pass
        except Exception as e:
            return f"[AnthropicAdapter SDK ERROR] {e}"

        try:
            import requests
            headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                       "content-type": "application/json"}
            payload = {"model": self.model, "max_tokens": 1024,
                       "messages": [{"role": "user", "content": prompt}]}
            if system_prompt:
                payload["system"] = system_prompt
            r = requests.post("https://api.anthropic.com/v1/messages",
                              json=payload, headers=headers, timeout=60)
            r.raise_for_status()
            return r.json()["content"][0]["text"].strip()
        except Exception as e:
            return f"[AnthropicAdapter REST ERROR] {e}"


# ─────────────────────────────────────────────────────────────────
# Mock (for testing / offline runs)
# ─────────────────────────────────────────────────────────────────

MOCK_RESPONSES = {
    "advaita": (
        "From the Advaita Vedānta perspective, the individual self (jīva) is ultimately "
        "non-different from Brahman. The apparent distinction is due to māyā — the cosmic "
        "illusory power. As Śaṅkarācārya teaches: 'Brahma satyaṃ jagan mithyā jīvo brahmaiva "
        "nāparaḥ' — Brahman alone is real; the world is apparent; the jīva is not other than "
        "Brahman. Therefore, liberation (mokṣa) consists in the direct realization of this "
        "non-dual identity, dissolving apparent karma."
    ),
    "dvaita": (
        "From the Dvaita Vedānta perspective of Madhvācārya, Brahman (Viṣṇu) and the jīva "
        "are eternally, fundamentally distinct — svarūpa-bheda. The jīva can never become "
        "Brahman. Liberation means dwelling in the presence of Viṣṇu in Vaikuṇṭha, enjoying "
        "bliss through His grace, while remaining distinct. Karma is erased only by the divine "
        "grace of Viṣṇu, not through self-effort alone."
    ),
    "nyaya": (
        "From the Nyāya-Vaiśeṣika perspective, the ātman is an eternal substance distinct "
        "from the body and mind. Through valid inference (anumāna) and testimony (śabda), we "
        "establish God's existence as the efficient cause of the universe. Dharma is known "
        "through Vedic injunctions. Liberation is the cessation of qualities (dharmas) of the "
        "ātman — a state of pure being without pain or pleasure."
    )
}

class MockAdapter(BaseLLMAdapter):
    """
    Returns school-appropriate canned responses.
    Used for testing, CI, and offline demos.
    """

    def __init__(self, school: str = "advaita"):
        self.school = school

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        base = MOCK_RESPONSES.get(self.school, MOCK_RESPONSES["advaita"])
        return f"[{self.school.upper()} — Mock Response]\n\n{base}\n\n[Query context: {prompt[:80]}...]"


# ─────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────

def get_adapter(backend: str = None, school: str = "advaita") -> BaseLLMAdapter:
    """
    Return the best available adapter.
    Priority: explicit param → LLM_BACKEND env var → auto-detect.

    Backends:
      gemini      → Google Gemini (set GEMINI_API_KEY)
      groq        → Groq cloud  (set GROQ_API_KEY)    [fastest free option]
      openai      → OpenAI      (set OPENAI_API_KEY)
      anthropic   → Claude      (set ANTHROPIC_API_KEY)
      ollama      → Local Ollama server
      huggingface → Local HuggingFace pipeline
      mock        → Offline testing
    """
    backend = backend or os.getenv("LLM_BACKEND", "auto")

    if backend == "mock":        return MockAdapter(school=school)
    if backend == "gemini":      return GeminiAdapter()
    if backend == "groq":        return GroqAdapter()
    if backend == "openai":      return OpenAIAdapter()
    if backend == "anthropic":   return AnthropicAdapter()
    if backend == "huggingface": return HuggingFaceAdapter()
    if backend == "ollama":      return OllamaAdapter()

    # auto-detect: priority order = Gemini > Groq > Ollama > OpenAI > Anthropic > Mock
    if os.getenv("GEMINI_API_KEY"):
        print("[LLM] Auto-detected: Gemini")
        return GeminiAdapter()
    if os.getenv("GROQ_API_KEY"):
        print("[LLM] Auto-detected: Groq")
        return GroqAdapter()
    # Try local Ollama
    ollama = OllamaAdapter()
    try:
        import requests as _req
        _req.get(f"{ollama.base_url}/api/tags", timeout=2)
        print("[LLM] Auto-detected: Ollama")
        return ollama
    except Exception:
        pass
    if os.getenv("OPENAI_API_KEY"):    return OpenAIAdapter()
    if os.getenv("ANTHROPIC_API_KEY"): return AnthropicAdapter()

    print("[LLM] No backend configured — using MockAdapter")
    print("[LLM] Set GEMINI_API_KEY or GROQ_API_KEY in .env for real LLM")
    return MockAdapter(school=school)
