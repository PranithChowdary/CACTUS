import json
import random
import os
from typing import List, Dict, Set

def prepare_data(input_path: str, output_dir: str, train_ratio=0.8, val_ratio=0.1):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    random.shuffle(data)
    
    # Build vocab
    vocab: Set[str] = set()
    for item in data:
        if isinstance(item.get('tool'), list):
            vocab.update(item['tool'])
        elif isinstance(item.get('tool'), str):
            vocab.add(item['tool'])
            
    vocab_list = sorted(list(vocab))
    vocab_list.append("DONE")
    
    # Save vocab
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab_list, f, indent=4)
    
    # Split
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Standardize format for Env (rename 'query' to 'Query', 'tool' to 'Tool' for compatibility)
    def standardize(subset):
        new_subset = []
        for item in subset:
            new_subset.append({
                "Query": item.get('query', ''),
                "Tool": ",".join(item['tool']) if isinstance(item['tool'], list) else item['tool']
            })
        return new_subset

    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(standardize(train_data), f, indent=4)
    with open(os.path.join(output_dir, 'val.json'), 'w') as f:
        json.dump(standardize(val_data), f, indent=4)
    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(standardize(test_data), f, indent=4)
    
    print(f"Data prepared: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test.")
    print(f"Vocab size: {len(vocab_list)}")

if __name__ == "__main__":
    prepare_data('data/Raw/multi_tool_query_golden.json', 'datasets')
