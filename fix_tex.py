import re

with open("Paper.tex", "r") as f:
    content = f.read()

# The table starts with \begin{table}[h] and has \caption{Main Experimental Results
table_pattern = r"(\\begin{table}\[h\]\n\\centering\n\\caption{Main Experimental Results.*?\\end{table})"

new_table = r"""\begin{table}[h]
\centering
\caption{Main Experimental Results on CACTUS Test Benchmark.}
\label{tab:main_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Agent Architecture} & \textbf{SEM ($\uparrow$)} & \textbf{F1-Score ($\uparrow$)} & \textbf{TL ($\downarrow$)} \\
\midrule
\textit{Non-Learning Baselines} \\
Uniform Random & 0.10 & 0.37 & 4.61 \\
Heuristic Rules & 0.00 & 0.33 & 4.78 \\
Dense Retrieval (\textit{all-MiniLM}) & 0.33 & 0.70 & 5.00 \\
\midrule
\textit{LLM Baselines (Zero-Shot)} \\
LLaMA-3-8B (ReAct) & - & - & - \\
GPT-4o-mini (Plan-and-Solve) & - & - & - \\
\midrule
\textit{RL Agents} \\
Behavioral Cloning (SFT) & - & - & - \\
PPO (CACTUS - Ours) & \textbf{0.67} & \textbf{0.85} & \textbf{3.08} \\
\bottomrule
\end{tabular}
\end{table}"""

content = re.sub(r"(\\begin{table}\[h\]\n\\centering\n\\caption{Main Experimental Results.*?\\end{table}\n)+", new_table + "\n", content, flags=re.DOTALL)

with open("Paper.tex", "w") as f:
    f.write(content)
