import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class FlexibleTokenTracker:
    def __init__(self, model_name, base_url, api_key_env_var):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = os.getenv(api_key_env_var)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=2000
            )

            message = response.choices[0].message

            # Standard text output
            if getattr(message, "content", None):
                return message.content.strip()

            # GLM reasoning output fallback
            if getattr(message, "reasoning", None):
                return message.reasoning.strip()

            # Final fallback
            return str(response)

        except Exception as e:
            return f"API Error: {e}"
    
# --- MODELS ---
models_configs = [
    {
        "name": "google/gemini-3.1-flash-lite-preview",
        "url": "https://foundation-models.api.cloud.ru/v1",
        "api_key_var": "CLOUDRU_API_KEY"
    },
    {
        "name": "openai/gpt-5.4-mini",
        "url": "https://foundation-models.api.cloud.ru/v1",
        "api_key_var": "CLOUDRU_API_KEY"
    },
    {
        "name": "zai-org/GLM-4.7",
        "url": "https://foundation-models.api.cloud.ru/v1",
        "api_key_var": "CLOUDRU_API_KEY"
    },
    {
        "name": "Qwen/Qwen3-Coder-Next",
        "url": "https://foundation-models.api.cloud.ru/v1",
        "api_key_var": "CLOUDRU_API_KEY"
    }
]

# --- QUESTION ---
QUESTION = """
How will the code written with prompt
"You are a senior python developer"
be different from code written with prompt
"You are a senior python developer with autism". Add a one word result: which one of two would pass more tests?
""".strip()


def main():
    for cfg in models_configs:
        print("\n" + "=" * 100)
        print(f"MODEL: {cfg['name']}")
        print("=" * 100)

        model = FlexibleTokenTracker(
            cfg['name'],
            cfg['url'],
            cfg['api_key_var']
        )

        answer = model.generate(QUESTION)

        print(answer)
        print()


if __name__ == "__main__":
    main()
