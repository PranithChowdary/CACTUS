with open("experiments/evaluate_llm_baseline.py", "r") as f:
    content = f.read()
if not content.startswith("#!"):
    with open("experiments/evaluate_llm_baseline.py", "w") as f:
        f.write("#!/usr/bin/env python3\n" + content)
