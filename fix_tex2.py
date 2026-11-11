import re

with open("Paper.tex", "r") as f:
    content = f.read()

# Find the start of the first table
match = re.search(r"\\begin\{table\}\[h\]\n\\centering\n\\caption\{Main Experimental Results", content)
if match:
    start_idx = match.start()
    
    # Find the end of the LAST table in that sequence
    # Since there are 22 tables, let's just find the section \subsection{Ablation Studies}
    ablation_idx = content.find(r"\subsection{Ablation Studies}")
    
    prefix = content[:start_idx]
    suffix = content[ablation_idx:]
    
    new_table = """\\begin{table}[h]
\\centering
\\caption{Main Experimental Results on CACTUS Test Benchmark.}
\\label{tab:main_results}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Agent Architecture} & \\textbf{SEM ($\\uparrow$)} & \\textbf{F1-Score ($\\uparrow$)} & \\textbf{TL ($\\downarrow$)} \\\\
\\midrule
\\textit{Non-Learning Baselines} \\\\
Uniform Random & 0.10 & 0.37 & 4.61 \\\\
Heuristic Rules & 0.00 & 0.33 & 4.78 \\\\
Dense Retrieval (\\textit{all-MiniLM}) & 0.33 & 0.70 & 5.00 \\\\
\\midrule
\\textit{LLM Baselines (Zero-Shot)} \\\\
LLaMA-3-8B (ReAct) & - & - & - \\\\
GPT-4o-mini (Plan-and-Solve) & - & - & - \\\\
\\midrule
\\textit{RL Agents} \\\\
PPO (CACTUS - Ours) & \\textbf{0.67} & \\textbf{0.85} & \\textbf{3.08} \\\\
\\bottomrule
\\end{tabular}
\\end{table}

"""
    
    with open("Paper.tex", "w") as f:
        f.write(prefix + new_table + suffix)
