#!/usr/bin/env python3
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Explicitly add the root directory to path to ensure no import errors, even though we don't import env here
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def evaluate():
    # We will use LLaMA-3.1-8B-Instruct which is already cached on your cluster!
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print(f"Loading {model_name} from local cache...")

    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto", local_files_only=True
    )
    
    with open("datasets/test.json") as f:
        test_data = json.load(f)
    with open("datasets/vocab.json") as f:
        vocab = json.load(f)
        
    tools_str = ", ".join([v for v in vocab if v != "DONE"])
    
    success_count = 0
    total = len(test_data)
    f1_sum = 0
    results = []

    print(f"Evaluating on {total} test queries...")
    
    for i, item in enumerate(test_data):
        query = item["Query"]
        ground_truth = set(item["Tool"].split(","))
        
        messages = [
            {"role": "system", "content": "You are a tool-selection agent. Output ONLY a comma-separated list of the exact tool names required from the provided list. Do not output any other text or explanation."},
            {"role": "user", "content": f"Available tools: {tools_str}\n\nUser Query: \"{query}\"\n\nWhich tools are needed?"}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Greedy decoding: deterministic and reproducible given fixed weights.
        # (temperature is meaningless under do_sample=False, so it's omitted
        # rather than passed-and-ignored, which previously triggered a
        # silent-no-op HF warning.)
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Parse output
        predicted_tools = [t.strip() for t in output_text.replace("`", "").split(",")]
        predicted_set = set([t for t in predicted_tools if t in vocab])
        
        # Metrics
        tp = len(predicted_set & ground_truth)
        fp = len(predicted_set - ground_truth)
        fn = len(ground_truth - predicted_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_sum += f1
        if predicted_set == ground_truth:
            success_count += 1
            
        print(f"[{i+1}/{total}] True: {ground_truth} | Pred: {predicted_set} | F1: {f1:.2f}")
        results.append({
            "index": i,
            "query": query,
            "ground_truth": sorted(ground_truth),
            "predicted": sorted(predicted_set),
            "f1": f1,
            "success": 1.0 if predicted_set == ground_truth else 0.0,
        })

    sem = success_count / total
    mean_f1 = f1_sum / total
    print(f"\n--- Final Results for {model_name} (Zero-Shot) ---")
    print(f"Success Rate (SEM): {sem:.2f}")
    print(f"F1-Score: {mean_f1:.2f}")

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/LLaMA-3.1-8B-Instruct.json", "w") as f:
        json.dump(results, f, indent=2)
    summary = {"LLaMA-3.1-8B-Instruct": {"success_mean": sem, "f1_mean": mean_f1, "n_examples": total}}
    with open("results/llm_baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved per-example results to results/raw/LLaMA-3.1-8B-Instruct.json "
          "and summary to results/llm_baseline_summary.json")


if __name__ == "__main__":
    evaluate()
