import json
import os
from huggingface_hub import InferenceClient

def evaluate():
    # Using LLaMA-3-8B-Instruct
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    client = InferenceClient(model_id, token=os.environ.get("HF_TOKEN"))
    
    with open("datasets/test.json") as f:
        test_data = json.load(f)
    with open("datasets/vocab.json") as f:
        vocab = json.load(f)
        
    tools_str = ", ".join([v for v in vocab if v != "DONE"])
    
    success_count = 0
    total = len(test_data)
    f1_sum = 0
    
    print(f"Evaluating {model_id} on {total} test queries...")
    
    for i, item in enumerate(test_data):
        query = item["Query"]
        ground_truth = set(item["Tool"].split(","))
        
        messages = [
            {"role": "system", "content": "You are a tool-selection agent. Output ONLY a comma-separated list of the exact tool names required from the provided list. Do not output any other text or explanation."},
            {"role": "user", "content": f"Available tools: {tools_str}\n\nUser Query: \"{query}\"\n\nWhich tools are needed?"}
        ]

        try:
            response = client.chat_completion(messages, max_tokens=30, temperature=0.01)
            output = response.choices[0].message.content
            
            # Parse comma-separated list
            predicted_tools = [t.strip() for t in output.replace("`", "").split(",")]
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
                
        except Exception as e:
            print(f"[{i+1}/{total}] Error calling HF API: {e}")
            continue
            
    final_success = success_count / total
    final_f1 = f1_sum / total
    print(f"\n--- Final Results for LLaMA-3-8B (Zero-Shot) ---")
    print(f"Success Rate (SEM): {final_success:.2f}")
    print(f"F1-Score: {final_f1:.2f}")

if __name__ == "__main__":
    if "HF_TOKEN" not in os.environ:
        print("ERROR: Please set your HF_TOKEN environment variable.")
        print("Run: export HF_TOKEN='your_token_here'")
    else:
        evaluate()
