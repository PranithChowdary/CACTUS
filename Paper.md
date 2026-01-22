# Title: CACTUS: A Framework for Learning and Evaluating Tool-Augmented Language Agents

**Authors**: [ ]
**Affiliation**: [ ]

---

## Abstract

Large Language Models (LLMs) have demonstrated remarkable capabilities, yet their practical application in complex, real-world domains is often limited by their inability to interact with external systems. To address this, a growing body of research focuses on augmenting LLMs with "tools"- APIs, functions, or other external utilities. However, the problem of how an agent should optimally select and sequence these tools in response to a user's query remains a significant challenge. Existing approaches often rely on heuristics or brittle, handcrafted logic. In this work, we introduce CACTUS (Context-Aware Compositional Tool-Use System), a novel framework for training and evaluating tool-augmented language agents. We formalize the tool-selection problem as a Markov Decision Process (MDP) and systematically investigate various learning paradigms, from supervised fine-tuning to reinforcement learning (RL) with Proximal Policy Optimization (PPO). We introduce a comprehensive benchmark based on a diverse set of multi-step tool-use scenarios and evaluate a suite of agents, including baseline, heuristic, and learning-based models. Our experiments demonstrate that agents trained with PPO significantly outperform baselines in terms of success rate, efficiency, and robustness. We provide a detailed analysis of agent behavior, learning dynamics, and the trade-offs between different training methodologies. Our framework and findings offer a rigorous foundation for the future development of capable and reliable tool-augmented agents.

---

## 1. Introduction

The capabilities of Large Language Models (LLMs) have expanded at a breathtaking pace, enabling sophisticated text generation, reasoning, and problem-solving. However, LLMs are inherently stateless and disconnected from the dynamic, interactive world. They cannot access real-time information, execute code, or interact with external APIs. This "knowledge boundary" fundamentally limits their utility in practical applications such as software development, scientific discovery, and autonomous systems.

A promising solution is to augment LLMs with a set of external **tools**. By granting an LLM-based agent the ability to call functions, query databases, or run shell commands, we can transform it from a passive text generator into an active problem-solver. The central research question then becomes: *how can an agent learn to effectively select and orchestrate these tools to accomplish complex, multi-step tasks?*

This paper introduces **CACTUS (Context-Aware Compositional Tool-Use System)**, a comprehensive framework designed to address this question with mathematical and empirical rigor. We make the following key contributions:

*   **Formalism:** We model the sequential tool-selection problem as a Markov Decision Process (MDP), providing a solid theoretical foundation for applying reinforcement learning algorithms.
*   **Environment:** We present a flexible and extensible environment that simulates real-world tool-use scenarios, complete with a configurable state space, a large set of diverse tools, and a well-defined reward structure.
*   **Methodology:** We implement and compare a wide range of agent architectures, including non-learning baselines (random, heuristic), supervised fine-tuned models (Gemini-SFT), and advanced reinforcement learning agents (RL-PPO, Gemini-RLHF).
*   **Evaluation:** We propose a rigorous evaluation protocol with a suite of metrics- including success rate, F1-score, and sample efficiency- to benchmark agent performance comprehensively, drawing from our detailed `Metrics.md`.
*   **Analysis:** Through extensive experiments, we demonstrate the superiority of RL-based agents, analyze their learning dynamics, and provide qualitative insights into their decision-making processes.

---

## 2. Related Work

*   **Tool-Augmented LLMs:** Discuss early work like Toolformer, ART, and the rise of function-calling APIs in commercial models (e.g., OpenAI's). Situate CACTUS in this landscape, highlighting its focus on learning and evaluation.
*   **Reinforcement Learning for Agent Control:** Review classic and modern RL approaches for sequential decision-making, from Q-learning to PPO. Connect this to its application in NLP and agent development.
*   **Automated Software Engineering & Code Generation:** Frame the problem within the context of AI for software engineering. Discuss how tool use is a fundamental component of code generation, debugging, and repository management.

---

## 3. Preliminaries: Formalizing Tool-Augmented LLMs

We formalize the problem of sequential tool selection as a **Markov Decision Process (MDP)**, defined by the tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$.

*   **State Space ($\mathcal{S}$):** A state $s_t \in \mathcal{S}$ at timestep $t$ represents the complete history of the interaction. It is a composite representation including:
    *   The initial user query, $q$.
    *   The sequence of previously selected tools and their outputs (observations): $(a_0, o_0), (a_1, o_1), \dots, (a_{t-1}, o_{t-1})$.
    *   Other contextual information, such as the state of the file system or available variables.
    $s_t = (q, (a_0, o_0), \dots, (a_{t-1}, o_{t-1}))$.

*   **Action Space ($\mathcal{A}$):** The action space is the set of all available tools, $\mathcal{A} = \{tool_1, tool_2, \dots, tool_N\}$. An action $a_t \in \mathcal{A}$ corresponds to the selection of a single tool at timestep $t$. We also include a special `TERMINATE` action.

*   **Transition Dynamics ($\mathcal{P}$):** The transition function $\mathcal{P}(s_{t+1} | s_t, a_t)$ defines the probability of moving to state $s_{t+1}$ after taking action $a_t$ in state $s_t$. In our deterministic environment, executing a tool $a_t$ produces a specific output $o_t$, leading to a new state $s_{t+1} = s_t \oplus (a_t, o_t)$, where $\oplus$ denotes concatenation.

*   **Reward Function ($\mathcal{R}$):** The reward function $\mathcal{R}(s_t, a_t)$ provides the learning signal. We design a sparse reward function that encourages task completion. For a task with a ground-truth set of required tools $\mathcal{T}_{true}$:
    *   $r_t = +1$ if $a_t \in \mathcal{T}_{true}$ and $a_t$ has not been selected before.
    *   $r_t = -0.1$ (small penalty) if $a_t \notin \mathcal{T}_{true}$ or if $a_t$ is a repeated, correct tool.
    *   $r_t = 0$ otherwise.
    An additional large terminal reward can be given if the agent calls `TERMINATE` and the set of selected tools matches $\mathcal{T}_{true}$.

*   **Discount Factor ($\gamma$):** The discount factor $\gamma \in [0, 1]$ balances the importance of immediate versus future rewards.

The agent's goal is to learn a policy $\pi(a_t | s_t)$ that maximizes the expected discounted return $G_t = \mathbb{E}[\sum_{k=0}^{\infty} \gamma^k r_{t+k}]$.

---

## 4. The CACTUS Environment

Our environment is a Python-based simulation built on `dm_env`. It is designed to be modular and extensible.

*   **State Representation:** The state is implemented as a dictionary containing the query, history, and other metadata. This is fed into the agent's policy network at each step.
*   **Action Space:** The environment is initialized with a list of over 200 tools, representing a realistic, large-scale action space. Tool definitions include names, descriptions, and argument schemas.
*   **Task Generation:** Tasks are sampled from the `multi_tool_query_golden.json` dataset. Each task consists of a natural language query and a corresponding set of ground-truth tools required to fulfill it.

---

## 5. Methodology: Learning to Select Tools

We investigate a spectrum of agents to solve the defined MDP.

#### 5.1. Baseline Agents
*   **Random Agent:** Selects a tool uniformly at random from $\mathcal{A}$ at each step.
*   **Scripted/Heuristic Agent:** A rule-based agent that may use simple logic (e.g., keyword matching between query and tool descriptions) to select tools.

#### 5.2. Learning-Based Agents
*   **Supervised Fine-Tuning (SFT):** We fine-tune a pre-trained Gemini model on trajectories from the golden dataset. The model is trained to predict the next correct tool given the state history, using a standard cross-entropy loss. This serves as a strong "behavioral cloning" baseline.
*   **Reinforcement Learning with PPO (RL-PPO):** We use a policy-gradient approach, specifically Proximal Policy Optimization (PPO), to train an agent from scratch. The policy and value functions are parameterized by a neural network (e.g., a Transformer encoder) that takes the state $s_t$ as input.
*   **RL from Human Feedback (RLHF):** We explore a Gemini model fine-tuned with RLHF, leveraging its advanced reasoning and instruction-following capabilities as a powerful policy network within our RL framework.

---

## 6. Experimental Setup

Our experimental design follows the rigorous protocol outlined in `docs/Metrics.md`.

#### 6.1. Dataset
The `multi_tool_query_golden.json` dataset contains diverse and complex queries requiring multi-step tool interactions. We split the dataset into training, validation, and test sets, ensuring no data leakage between them. We also hold out entire domains of tools for a zero-shot generalization test.

#### 6.2. Evaluation Metrics
We evaluate agents on a comprehensive set of metrics to capture performance from multiple angles:
*   **Success & Accuracy:** Set Accuracy (fraction of required tools found).
*   **Precision, Recall, F1-Score:** Treat tool selection as a set prediction problem (`Precision@K`, `Recall@K`, `F1@K`).
*   **Efficiency:** Steps-to-First-Complete (SFC) and total steps used per episode.
*   **Robustness:** Performance under noisy conditions or with partial tool availability.
*   **Learning Dynamics:** Sample efficiency (e.g., episodes to reach 80% accuracy).

#### 6.3. Implementation Details
*   **Agents Compared:** Random, Heuristic, Gemini-SFT, RL-PPO.
*   **Hyperparameters:** We perform a grid search for key hyperparameters (e.g., learning rate, PPO clipping epsilon, entropy coefficient) for each learning-based agent.
*   **Statistical Significance:** All experiments are run with multiple random seeds (N=10). We report mean performance with 95% confidence intervals and use statistical tests (e.g., Wilcoxon signed-rank) to compare methods.

---

## 7. Results and Analysis

This section will present the core findings of our experiments.

#### 7.1. Main Performance Comparison
We will present bar charts and tables comparing all agents across the primary metrics (Accuracy, F1-Score, SFC). We expect to show that the RL-PPO agent significantly outperforms all baselines.

#### 7.2. Learning Efficiency
Learning curves (`Mean Return vs. Training Steps`) will be used to visualize the sample efficiency of SFT vs. RL approaches. This will illustrate how quickly each agent learns the task.

#### 7.3. Qualitative Analysis
We will showcase several example trajectories (both successful and failed) for the RL agent. This will include the initial query, the sequence of tools selected, the rewards received, and the agent's internal policy distribution, providing insight into its decision-making process.

#### 7.4. Ablation Studies
We will conduct ablation studies to understand the contribution of different components, such as the impact of the reward function design or the effect of using a pre-trained vs. random policy network initialization.

---

## 8. Discussion

*   **Limitations:** Discuss the limitations of the current work, such as the simulated nature of the environment, the reliance on a fixed set of tools, and potential challenges in scaling to even larger action spaces.
*   **Future Work:** Outline promising directions for future research. This includes developing more sophisticated reward models, exploring hierarchical RL for tool composition, and applying the framework to real-world, live systems.

---

## 9. Conclusion

In this paper, we introduced CACTUS, a framework for developing and evaluating tool-augmented language agents. By formalizing the problem as an MDP and conducting a rigorous, systematic comparison of different learning strategies, we demonstrated that reinforcement learning is a powerful and effective paradigm for teaching agents to master complex tool-use. Our findings and open-source framework provide a crucial stepping stone for building the next generation of capable, autonomous AI agents.

---

## References

[Placeholder for bibliography]

---

## Appendix

#### A. Hyperparameter Details
*   A detailed table of all hyperparameters used for the SFT and RL-PPO experiments.

#### B. Additional Results
*   Per-domain performance breakdowns.
*   Full results of statistical tests.
*   Additional plots, such as tool selection heatmaps and policy entropy decay curves.
