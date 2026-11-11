with open("Paper.tex", "r") as f:
    content = f.read()

import re
content = content.replace("\\begin{table}[h]\n\\centering\n\\begin{table}[h]\n\\centering", "\\begin{table}[h]\n\\centering")
content = content.replace("Table \\ref{tab:main_results} defines the rigorous comparative framework for evaluating CACTUS agents.\n\n\\begin{table}[h]", "Table \\ref{tab:main_results} defines the rigorous comparative framework for evaluating CACTUS agents.\n\n\\begin{table}[h]")

with open("Paper.tex", "w") as f:
    f.write(content)
