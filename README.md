# Recursive Reasoning Framework
### *A Scientific-Method-as-a-Service Harness for LLM Agents*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22214013.svg)](https://doi.org/10.5281/zenodo.22214013)

## 1. Abstract
The Recursive Reasoning Framework (RRF) is a domain-agnostic testing harness designed to move Large Language Model (LLM) agents away from "lucky guessing" and toward systematic discovery. Unlike standard evaluation harnesses that rely on large context windows or simple success/fail scalars, RRF enforces a **discrete cognitive layer** through a loop of hypothesis generation, deterministic backtesting, and cross-task memory persistence.

By decoupling environment logs from the prompt and utilizing an atomic JSON memory loop, RRF enables **Zero-Shot Knowledge Transfer**. An agent can solve a complex task using logical deductions inherited from past execution histories without requiring fine-tuning or prompt-bloat.

---

## 2. The Core Philosophy
* **Token Efficiency**: Instead of feeding entire execution histories into the LLM context, RRF provides a local API. The agent only processes the current state, using deterministic Python scripts to query history locally.
* **Deterministic Rigor**: Agents are restricted from testing hypotheses in the live environment if those hypotheses can be falsified by existing logs.
* **Mechanism Elucidation**: The harness mandates that agents state a explicit *predicted outcome* before acting, turning every action into a formal scientific experiment.
* **Recursive Meta-Learning**: Discovered knowledge is stored in a structured schema, allowing the framework to serve as a meta-learning complex for self-directed exploration.

---

## 3. Case Studies
### Case Study 1: The Cryptic State-Machine (Inherited Reasoning)

To validate the framework, a **Cryptic State-Machine** environment was implemented where rules are hidden, deterministic, and sequential.

* **Task 1 (`task_1_blue`)**: Discover the sequence to turn an indicator Blue.
* *Finding*: The agent conducted exploratory button presses, identified `PRESS_A` as a reset, and eventually deduced that `PRESS_B` → `PRESS_C` = **Blue**.
* *Consolidation*: The agent wrote this rule to `global_memory.json` with "High" confidence.

* **Task 2 (`task_2_green`)**: Discover the sequence to turn the indicator Green.
* *Finding*: The agent **inherited the Blue rule**. Bypassing all trial-and-error for the first half of the task, it immediately executed `B → C`, and successfully isolated `PRESS_D` → `PRESS_E` as the final transition to **Green**.

**Results (from `results.json`)**

| Task | Solved | Actions | Learning Type |
| --- | --- | --- | --- |
| `task_1_blue` | ✅ | 4 | Exploratory Discovery |
| `task_2_green` | ✅ | 5 | **Inherited Reasoning** |

*Analysis: The agent solved a deeper logic chain in nearly the same number of steps as the simpler task, proving that deductive reasoning was successfully transferred through the memory loop.*

### Case Study 2: Sokoban Microban (Emergent Algorithmic Search)

To verify that the dynamic plugin system (`core/harness.py`) requires **no core edits**, a single file (`plugins/sokoban.py`) was introduced for Microban Level 1 (David Skinner, 2000).

* **Agent Trace (`logs/microban_1_agent.log`)**: The agent generated a 33-move, BFS-optimal path (`DURATION 63s`). Every `POST /step` API call included a `predicted_outcome` that was successfully validated against `logs/microban_1_env.log` prior to execution.
* **Cross-Level Knowledge (memory/global_memory.json)**: After Level 1, the agent recorded seven "high" confidence rules. Five of these were generic, transferable physical rules (e.g., walls block movement, illegal push definitions, win condition `boxes⊆goals`).
* **Rule Generalization**: When exposed to a new level (`microban_2`), the agent loaded `global_memory.json`, verified the generic rules with a single action, and inherited the physical mechanics. Subsequent levels became easier because only the layout required resolution, eliminating the need for physics rediscovery.

**Emergence of BFS-Optimal Pathing (Not Hard-Coded)**

The system prompt (`SKILLS.template.md`) never explicitly mentions Breadth-First Search. It merely dictates: *Write and execute Python to test hypothesis against logs before acting.*

Because exhaustive search is required to test spatial hypotheses reliably, the LLM autonomously inferred the need for a BFS algorithm. It wrote the necessary Python code locally, resulting in optimal paths for both levels (33 steps for Level 1; 16 steps for Level 2) through self-directed algorithm generation rather than hard-coded heuristics.

| Task | Boxes | Steps | Duration | Type |
| --- | --- | --- | --- | --- |
| `microban_1` | 2 | 33 | 63s | Exploratory (mechanics discovery) |
| `microban_2` | 3 | 16 | 102s | Inherited (physics reused) |

---

## 4. How to Run

### Prerequisites

* Python 3.12+
* An LLM provider accessible via OpenCode (the default agent driver for this framework).

### Installation

```bash
git clone https://github.com/[repository-url]/rrf.git
cd rrf
pip install -r requirements.txt

```

### Execution Examples

```bash
# State-machine demo (runs two sequential tasks):
bash run-engine.sh  

# Sokoban single-level discovery:
python3 -m core.harness --env sokoban --tasks microban_1

# Sokoban cross-level transfer:
python3 -m core.harness --env sokoban --tasks microban_1 microban_2

```

---

## 5. Extending the Harness

RRF is designed for seamless extensibility. Any Python file placed in the `plugins/` directory containing a `BaseEnvironment` subclass is auto-discovered. No harness edits or manual registrations are required.

### Creating a Plugin

1. **Define the Logic**: Create `plugins/custom_env.py`.
2. **Inherit & Implement**:

```python
from core.environment import BaseEnvironment

class CustomEnv(BaseEnvironment):
    domain = "custom_domain"  # Utilized for memory filtering
    description = "One-line description for --list-envs"

    def get_valid_actions(self):
        return ["ACTION_1", "ACTION_2"]

    def start(self, task_id):
        return {"status": "init"}

    def step(self, action_payload):
        # Domain logic implementation
        return observation, is_done, is_win

    def get_observation_schema(self):
        return {"type": "object"}  # Optional JSON-schema validation

```

3. **Execute**: The harness dynamically loads the new environment:
```bash
python -m core.harness --list-envs
python -m core.harness --env custom_env --tasks task_1

```

See `plugins/counter_env.py` for a minimal tutoring example, or `plugins/state_machine.py` for the original proof-of-concept.

### Cross-Domain Applicability

Because the interaction contract is limited to `get_valid_actions`, `start`, and `step`, RRF functions effectively as a general-purpose testing harness for:

* **Puzzles & Games**: Grid-worlds, text adventures.
* **Tool-Use & API Testing**: Mock REST/DB endpoints where `step` validates schema compliance.
* **Education & Tutoring**: Arithmetic and logic chain discovery.
* **Research**: Multi-domain filtering and open-ended exploration leveraging the `global_memory.json` persistence layer.

### Stochastic Discovery & AI/Human Equivalence

**Stochastic rule inference is fully tractable** provided the underlying mechanism is discoverable. For example, in a 6-sided die plugin (`plugins/die_env.py`), the `step()` function returns a sampled integer along with a `variance_proxy`. This enables the LLM to run `scipy.stats.chisquare` against the execution logs, fit a uniform distribution, and write a "medium" or "low" confidence rule to the global memory.

Because predicting a single roll cannot exceed a 1/6 probability, the agent relies on contradiction-handling protocols to downgrade confidence rather than asserting absolute truth. The only necessary adaptation is that the deterministic falsification requirement in `SKILLS.template.md` flexes to allow Bayesian fitting or K-S testing, all handled entirely on the plugin side.

**AI and Human operators share the same theoretical ceiling under this contract:**

* Both require prior knowledge of algorithmic mechanisms (without it, "intuition" is merely guessing).
* Pure randomness remains uncomputable for both.
* The primary difference lies in the auditability: an LLM provides an explicit, reproducible backtest loop, whereas human logic is often opaque.

### Autonomy Boundaries

* **Inner-Loop (LLM Autonomous)**: The `TaskSession` mechanically logs frames, while the agent executes the hypothesize → backtest → predict → update sequence entirely without human intervention.
* **Outer-Loop (Human-Directed)**: Task selection, stop conditions, and ultimate win rules are defined externally by the operator. The agent operates autonomously within the boundaries of the framework, but does not self-spawn or generate its own ultimate objectives.

---

## 6. Guidance for New Users

1. **Monitor the Agent's Logic**: Utilize `tail -f logs/*_agent.log` to watch the agent actively write and execute Python code to verify its hypotheses in real-time.
2. **Review the Memory Artifacts**: Inspect `memory/global_memory.json` after an execution. This file serves as the functional backbone of the agent's learned reasoning.
3. **Adjust the Rigor**: The `SKILLS.template.md` file defines the agent's behavioral workflow. Modifying this template allows for stricter or looser adherence to the scientific method.

### Verifying System Modularity

To confirm that adding a plugin requires no edits to the core execution engine, utilize the following commands:

```bash
ls -lh logs/                            # Review harness logs
cat logs/microban_1_env.log             # Verify Sokoban state traces
cat memory/global_memory.json | python3 -m json.tool  # Inspect domain filtering
python3 -m core.harness --list-envs     # Verify auto-discovery system
git diff core/                          # Ensure this returns empty (proves modularity)

```

---

## 7. Future Directions

* **Multi-Domain Filtering**: Training agents to filter the JSON memory for relevant rules based strictly on active domain tags.
* **Contradiction Handling**: Implementing native logic for agents to programmatically downgrade confidence in stored rules if an environment's laws change (Non-Stationary Environments).
* **Self-Recursive Exploration**: Setting the agent on an open-ended directive to discover all possible rules within a highly complex, unknown environment.

---

## 8. Capability Recursion vs. Self-Recursion & Amplifier Framing

* **Capability Recursion**: The dynamic bridge between the LLM and the Python ecosystem. During log backtesting, the LLM can invoke any standard or third-party library (`scipy`, `sympy`, or even subprocesses to Lean/Fortran). This is bounded by *knowability* (the LLM must understand the algorithmic mechanism) and *testability* (the plugin must expose the necessary variables). Obscure algorithms are not spontaneously invented; rather, known computational tools are systematically applied.
* **Self-Recursion**: An aspirational state where an agent defines its own hypotheses, priorities, and stop conditions. Current implementations require proximate motivation originating from a human operator.
* **The Amplification Effect**: Without RRF, human operators must manually code, execute, and debug algorithms like BFS or chi-square tests. With RRF, the LLM autonomously writes the algorithm, validates the `predicted_outcome`, and logs the resulting rule with a full audit trail. The framework acts as a large multiplier for human-directed inquiry, accelerating rigorous discovery across domains.

---

### License

* This project is licensed under the MIT License.

### Citation

* Friedman, R. (2026) *Recursive Reasoning Framework* (v1.1). Zenodo. https://doi.org/10.5281/zenodo.22214013

### Acknowledgments

* The conceptual development of the methodology and codebase benefited from discussions and iterative refinement with AI language models, Gemini 3.1 Pro (Google) and Muse Spark 1.2 (Meta).