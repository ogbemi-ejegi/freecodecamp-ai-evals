import os
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM

load_dotenv()

class OpenRouterJudge(DeepEvalBaseLLM):
    """
    Custom DeepEval LLM-as-a-Judge wrapper powered by OpenRouter.
    Includes a 60s socket timeout and automatic retries to prevent stalled connections.
    """
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("JUDGE_MODEL", "deepseek/deepseek-v4-pro-0813")
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OPENROUTER_API_KEY in .env file!")

        self._sync_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            timeout=60.0,
            max_retries=2,
        )
        self._async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            timeout=60.0,
            max_retries=2,
        )

    def load_model(self):
        return self._sync_client

    def _clean_response(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        return cleaned

    def generate(self, prompt: str) -> str:
        response = self._sync_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        return self._clean_response(content)

    async def a_generate(self, prompt: str) -> str:
        response = await self._async_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        return self._clean_response(content)

    def get_model_name(self) -> str:
        return f"OpenRouter ({self.model_name})"