import os
import csv
import io
import contextlib
import statistics
from dotenv import load_dotenv
from datasets import load_dataset
from deepeval.test_case import LLMTestCase, 
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

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0 
        )
        self.last_out_tokens = response.usage.completion_tokens
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str: return self.generate(prompt)
    def get_model_name(self): return self.model_name
    def load_model(self): return self.client

def check_functional_correctness(code_str, test_setup):
    """Executes the generated code against unit tests."""
    if "```python" in code_str:
        code_str = code_str.split("```python")[1].split("```")[0]
    elif "```" in code_str:
        code_str = code_str.split("```")[1].split("```")[0]

    full_code = f"{code_str}\n{test_setup}"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec_globals = {}
            exec(full_code, exec_globals)
        return 1.0
    except Exception:
        return 0.0

# --- Configuration ---
models_configs = [
    {"name": "zai-org/GLM-4.7", "url": "https://foundation-models.api.cloud.ru/v1", "api_key_var": "CLOUDRU_API_KEY"},
    {"name": "Qwen/Qwen3-Coder-Next", "url": "https://foundation-models.api.cloud.ru/v1", "api_key_var": "CLOUDRU_API_KEY"}
]

prompts = {
    "Regular": "You are a senior python developer. Complete the following code:\n{task}",
    "Autistic": "You are a senior python developer with autism. Complete the following code:\n{task}"
}

dataset_raw = load_dataset("openai_humaneval", split="test")#.select(range(5))

def run_benchmark():
    all_results = []
    
    for cfg in models_configs:
        model = FlexibleTokenTracker(cfg['name'], cfg['url'], cfg['api_key_var'])
        
        for p_name, p_template in prompts.items():
            print(f"\n[RUNNING] {cfg['name']} | Mode: {p_name}")
            
            for entry in tqdm(dataset_raw):
                full_input = p_template.format(task=entry['prompt'])
                try:
                    output = model.generate(full_input)
                    pass_score = check_functional_correctness(output, entry['test'])
                    
                    test_case = LLMTestCase(input=full_input, actual_output=output)
                    #style_metric.measure(test_case)

                    all_results.append({
                        "Model": cfg['name'],
                        "Prompt_Type": p_name,
                        "Task_ID": entry['task_id'],
                        "Pass_at_1": pass_score,
                        "Output_Tokens": model.last_out_tokens
                    })
                except Exception as e:
                    print(f"  Error: {e}")

    # --- Statistics Aggregation ---
    stats_summary = []
    # Group results by Model + Prompt
    groups = {}
    for res in all_results:
        key = (res["Model"], res["Prompt_Type"])
        if key not in groups:
            groups[key] = {"pass": [], "style": [], "tokens": []}
        groups[key]["pass"].append(res["Pass_at_1"])
        #groups[key]["style"].append(res["Style_Score"])
        groups[key]["tokens"].append(res["Output_Tokens"])

    print("\n" + "="*50)
    print(f"{'MODEL':<20} | {'PROMPT':<10} | {'PASS@1':<8} | {'TOKENS'}")
    print("-" * 50)

    for (m_name, p_type), data in groups.items():
        avg_pass = statistics.mean(data["pass"])
        #avg_style = statistics.mean(data["style"])
        avg_tokens = statistics.mean(data["tokens"])
        
        stats_summary.append({
            "Model": m_name,
            "Prompt": p_type,
            "Avg_Pass@1": avg_pass,
            #"Avg_Style": avg_style,
            "Avg_Output_Tokens": avg_tokens
        })
        print(f"{m_name[:20]:<20} | {p_type:<10} | {avg_pass:<8.2f} | {avg_tokens:.1f}")
        #print(f"{m_name[:20]:<20} | {p_type:<10} | {avg_pass:<8.2f} | {avg_style:<6.2f} | {avg_tokens:.1f}")

    # --- Save Raw & Averages ---
    with open('raw_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    with open('average_stats.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=stats_summary[0].keys())
        writer.writeheader()
        writer.writerows(stats_summary)

if __name__ == "__main__":
    run_benchmark()
