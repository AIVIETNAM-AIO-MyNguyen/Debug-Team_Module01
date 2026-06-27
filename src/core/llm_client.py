import os
import re
import requests
import logging

logger = logging.getLogger("rag_bench.llm_client")

def load_env(env_path=".env"):
    """Loads environment variables from a .env file into os.environ if it exists."""
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip()
                        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                            val = val[1:-1]
                        os.environ[key] = val

# Load env variables during import
load_env()

class PuterMiniMaxClient:
    """OpenAI-compatible client wrapper for Puter's free/unlimited MiniMax API."""
    
    def __init__(self, model: str = "minimax/minimax-m2.5"):
        self.model = model
        self.base_url = "https://api.puter.com/puterai/openai/v1/chat/completions"
        self._warned = False

    @property
    def token(self) -> str:
        return os.environ.get("PUTER_TOKEN") or os.environ.get("PUTER_API_KEY")

    def __call__(self, prompt: str) -> str:
        """Allows direct invocation matching mock_llm_client's signature."""
        return self.generate(prompt)

    def generate(self, prompt: str) -> str:
        token = self.token
        if not token:
            raise ValueError(
                "PUTER_TOKEN or PUTER_API_KEY environment variable is not set. "
                "Real LLM client requires an active Puter token."
            )

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }

        try:
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                # Clean out <think>...</think> blocks if model tries to output reasoning
                cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return cleaned_content
            else:
                raise RuntimeError(
                    f"Puter API returned error status {response.status_code}: {response.text}"
                )
        except Exception as e:
            logger.error(f"Error calling Puter API: {e}")
            raise

class GroqClient:
    """OpenAI-compatible client wrapper for Groq's API."""
    
    def __init__(self, model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    @property
    def token(self) -> str:
        return os.environ.get("GROQ_API_KEY")

    def __call__(self, prompt: str) -> str:
        """Allows direct invocation matching client signatures."""
        return self.generate(prompt)

    def generate(self, prompt: str) -> str:
        token = self.token
        if not token:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Real LLM client requires an active Groq API key."
            )

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }

        import time
        max_retries = 8
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.base_url, json=payload, headers=headers, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    # Clean out <think>...</think> blocks if the model tries to output reasoning
                    cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                    return cleaned_content
                elif response.status_code == 429:
                    error_text = response.text
                    # Try to extract wait time from the Groq error message
                    # Example: "Please try again in 4.2075s."
                    wait_match = re.search(r"try again in (\d+\.?\d*)s", error_text)
                    if wait_match:
                        sleep_time = float(wait_match.group(1)) + 0.5
                    else:
                        sleep_time = base_delay * (2 ** attempt)
                    
                    logger.warning(
                        f"Groq API Rate Limit (429) hit. Attempt {attempt + 1}/{max_retries}. "
                        f"Sleeping for {sleep_time:.2f}s before retrying. Error details: {error_text}"
                    )
                    time.sleep(sleep_time)
                else:
                    raise RuntimeError(
                        f"Groq API returned error status {response.status_code}: {response.text}"
                    )
            except Exception as e:
                # If we get a requests/connection exception and we have retries left, we back off and retry
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    logger.warning(f"Error calling Groq API: {e}. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Error calling Groq API on final attempt: {e}")
                    raise
        raise RuntimeError("Failed to generate response from Groq API after maximum retries.")


class OllamaClient:
    """Client wrapper for local Ollama instance."""
    
    def __init__(self, model: str = "gemma4:e4b"):
        self.model = model
        self.base_url = "http://localhost:11434/api/generate"

    def __call__(self, prompt: str) -> str:
        """Allows direct invocation matching client signatures."""
        return self.generate(prompt)

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.base_url, json=payload, timeout=300)
            if response.status_code == 200:
                data = response.json()
                content = data.get("response", "").strip()
                cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return cleaned_content
            else:
                raise RuntimeError(
                    f"Ollama API returned error status {response.status_code}: {response.text}"
                )
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            raise



