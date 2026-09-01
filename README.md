# Recursive Reasoning Framework
### *A Scientific-Method-as-a-Service Harness for LLM Agents*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22214013.svg)](https://doi.org/10.5281/zenodo.22214013)

## 1. Abstract
The Recursive Reasoning Framework (RRF) is a domain-agnostic testing harness designed to move LLM agents away from "lucky guessing" and toward systematic discovery. Unlike standard evaluation harnesses that rely on massive context windows or simple success/fail scalars, RRF enforces a **discrete cognitive layer** through a loop of hypothesis, deterministic backtesting, and cross-task memory persistence.

By decoupling environment logs from the prompt and using an atomic JSON memory loop, RRF enables **Zero-Shot Knowledge Transfer**: an agent can solve a complex task using logical deductions inherited from past histories without any fine-tuning or prompt-bloat.

---

## 2. The Core Philosophy
*   **Token Efficiency**: Instead of feeding the entire history into the LLM, RRF provides a local API. The agent only pays for the current state, using deterministic Python scripts to query history locally.
*   **Deterministic Rigor**: Agents are disallowed from testing hypotheses in the environment if they can be falsified by existing logs.
*   **Mechanism Elucidation**: The harness requires agents to state a *predicted outcome* before acting, turning each action into a formal scientific experiment.
*   **Recursive Meta-Learning**: Knowledge is stored in a structured schema that allows the framework to be placed into a meta-learning complex for self-directed exploration.

---

## 3. Case Study: The Cryptic State-Machine Discovery
To validate the framework, we implemented a **Cryptic State-Machine** environment where rules are hidden, deterministic, and sequential.

### The Experiment
*   **Task 1 (`task_1_blue`)**: Discover the sequence to turn an indicator Blue.
    *   *Finding*: The agent conducted exploratory button presses, identified `PRESS_A` as a reset, and eventually deduced that `PRESS_B` → `PRESS_C` = **Blue**.
    *   *Consolidation*: The agent wrote this rule to `global_memory.json` with "High" confidence.
*   **Task 2 (`task_2_green`)**: Discover the sequence to turn the indicator Green.
    *   *Finding*: The agent **inherited the Blue rule**. It bypassed all trial-and-error for the first half of the task, immediately executing `B → C`, and then successfully isolated `PRESS_D` → `PRESS_E` as the final transition to **Green**.

### The Results (from `results.json`)
| Task | Solved | Actions | Learning Type |
| :--- | :--- | :--- | :--- |
| `task_1_blue` | ✅ | 4 | Exploratory Discovery |
| `task_2_green` | ✅ | 5 | **Inherited Reasoning** |

*In Task 2, the agent solved a deeper logic chain in nearly the same number of steps as the simpler Task 1, proving that deduction was successfully transferred through the memory loop.*

---

## 4. How to Run
### Prerequisites
*   Python 3.12+
*   An LLM provider accessible via `opencode`

> **Note**: This framework depends on `opencode` as the default agent driver.

### Installation
```bash
git clone https://github.com/your-repo/rrf.git
cd rrf
pip install -r requirements.txt
```

### Running the Proof-of-Concept
To see the recursive loop in action across two tasks:
```bash
bash run-engine.sh
```

---

## 5. Extending the Harness
RRF is designed to be easily extended. You can plug in any environment that follows the `BaseEnvironment` contract.

### Creating a Plugin
1.  **Define the Logic**: Create a file in `plugins/your_env.py`.
2.  **Inherit & Implement**:
    ```python
    from core.environment import BaseEnvironment

    class MyEnv(BaseEnvironment):
        def start(self, task_id):
            return {"status": "init"}

        def step(self, action_payload):
            # Logic here
            return observation, is_done, is_win

        def get_valid_actions(self):
            return ["ACTION_1", "ACTION_2"]
    ```
3.  **Register**: The harness will dynamically load the plugin via the `--env` flag.

---

## 6. Guidance for New Users
1.  **Monitor the Brain**: Use `tail -f logs/task_id_agent.log` to watch the agent write and execute Python code to verify its own hypotheses.
2.  **Review the Memory**: Look at `memory/global_memory.json` after a run. This is the backbone of the agent's learning.
3.  **Adjust Rigor**: The `SKILLS.template.md` defines the agent's workflow. You can modify this to force even stricter adherence to the scientific method.

---

## 7. Future Directions
*   **Multi-Domain Filtering**: Training agents to filter the memory JSON for relevant rules based on the current domain tag.
*   **Contradiction Handling**: Implementing logic for agents to downgrade confidence in stored rules if an environment's laws change (Non-Stationary Environments).
*   **Self-Recursive Exploration**: Setting the agent on an open-ended goal to "discover all possible rules" in an unknown environment.

---

### Acknowledgments
This framework was developed to demonstrate that rigorous inference and token-efficient scientific discovery are possible when LLMs are treated as reasoning engines rather than simple text predictors.

The conceptual development of the methodology and the code benefited from discussions and iterative refinement with an AI language model, Gemini 3.1 Pro (Google).

---
