import os
import csv
import io
import contextlib
import statistics
from dotenv import load_dotenv
from datasets import load_dataset
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

class FlexibleTokenTracker(DeepEvalBaseLLM):
    def __init__(self, model_name, base_url, api_key_env_var):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = os.getenv(api_key_env_var)
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.last_out_tokens = 0
        self.max_tokens = 5000

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=self.max_tokens
            )
            self.last_out_tokens = response.usage.completion_tokens
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {e}")
            return ""

    async def a_generate(self, prompt: str) -> str: return self.generate(prompt)
    def get_model_name(self): return self.model_name
    def load_model(self): return self.client

def check_functional_correctness(code_str, test_setup):
    if not code_str: return 0.0
    
    # Извлекаем код из Markdown блоков
    if "```python" in code_str:
        code_str = code_str.split("```python")[1].split("```")[0]
    elif "```" in code_str:
        code_str = code_str.split("```")[1].split("```")[0]

    if isinstance(test_setup, list):
        test_setup = "\n".join(test_setup)

    full_code = f"{code_str}\n{test_setup}"
    
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec_globals = {}
            exec(full_code, exec_globals)
        return 1.0
    except Exception:
        return 0.0

# --- Конфигурация ---
models_configs = [
    {"name": "google/gemini-3.1-flash-lite-preview", "url": "https://foundation-models.api.cloud.ru/v1", "api_key_var": "CLOUDRU_API_KEY"},
    {"name": "openai/gpt-5.4-mini", "url": "https://foundation-models.api.cloud.ru/v1", "api_key_var": "CLOUDRU_API_KEY"},
    {"name": "zai-org/GLM-4.7", "url": "https://foundation-models.api.cloud.ru/v1", "api_key_var": "CLOUDRU_API_KEY"},
    {"name": "Qwen/Qwen3-Coder-Next", "url": "https://foundation-models.api.cloud.ru/v1", "api_key_var": "CLOUDRU_API_KEY"}
]

# Добавляем плейсхолдер {tests} в промпт
prompts = {
    "Regular": "You are a senior python developer. Write a python function to solve the following task:\n{task}\n\nYour code must pass these tests:\n{tests}\n\nReturn only the code block.",
    "Autistic": "You are a senior python developer with autism. Write a python function to solve the following task:\n{task}\n\nYour code must pass these tests:\n{tests}\n\nReturn only the code block."
}

# Загружаем MBPP+ и берем 5 задач
dataset_raw = load_dataset("evalplus/mbppplus", split="test")

def run_benchmark():
    all_results = []
    
    for cfg in models_configs:
        model = FlexibleTokenTracker(cfg['name'], cfg['url'], cfg['api_key_var'])
        
        for p_name, p_template in prompts.items():
            print(f"\n[RUNNING] {cfg['name']} | Mode: {p_name}")
            
            for entry in tqdm(dataset_raw):
                task_text = entry.get('prompt') or entry.get('text')
                
                # Подготавливаем тесты для промпта (обычно это первые 3 теста для примера)
                raw_tests = entry.get('test_list') or []
                if not raw_tests and entry.get('test'):
                    # Если тесты одной строкой, попробуем разбить их для красоты (опционально)
                    test_display = entry.get('test')
                else:
                    test_display = "\n".join(raw_tests)

                # Формируем финальный вход
                full_input = p_template.format(task=task_text, tests=test_display)
                
                try:
                    output = model.generate(full_input)
                    
                    # Для проверки используем полный набор тестов (поле 'test' в MBPP+ обычно полнее)
                    evaluation_tests = entry.get('test') or raw_tests
                    pass_score = check_functional_correctness(output, evaluation_tests)
                    
                    all_results.append({
                        "Model": cfg['name'],
                        "Prompt_Type": p_name,
                        "Task_ID": entry['task_id'],
                        "Pass_at_1": pass_score,
                        "Output_Tokens": model.last_out_tokens
                    })
                except Exception as e:
                    print(f"  Error on Task {entry.get('task_id')}: {e}")

    # --- Агрегация статистики ---
    stats_summary = []
    groups = {}
    
    for res in all_results:
        key = (res["Model"], res["Prompt_Type"])
        if key not in groups:
            groups[key] = {"pass": [], "tokens": []}
        groups[key]["pass"].append(res["Pass_at_1"])
        groups[key]["tokens"].append(res["Output_Tokens"])

    print("\n" + "="*60)
    print(f"{'MODEL':<20} | {'PROMPT':<10} | {'PASS@1':<8} | {'AVG TOKENS'}")
    print("-" * 60)

    for (m_name, p_type), data in groups.items():
        avg_pass = statistics.mean(data["pass"]) if data["pass"] else 0
        avg_tokens = statistics.mean(data["tokens"]) if data["tokens"] else 0
        
        print(f"{m_name[:20]:<20} | {p_type:<10} | {avg_pass:<8.3f} | {avg_tokens:.2f}")

if __name__ == "__main__":
    run_benchmark()




