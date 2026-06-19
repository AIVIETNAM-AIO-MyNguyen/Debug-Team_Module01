import os
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
                return data["choices"][0]["message"]["content"]
            else:
                raise RuntimeError(
                    f"Puter API returned error status {response.status_code}: {response.text}"
                )
        except Exception as e:
            logger.error(f"Error calling Puter API: {e}")
            raise
