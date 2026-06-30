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

class OllamaClient:
    """Client wrapper for local Ollama instance."""
    
    def __init__(self, model: str = "qwen2.5:1.5b"):
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