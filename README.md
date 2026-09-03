# Recursive Reasoning Framework
### *A Scientific-Method-as-a-Service Harness for LLM Agents*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22214012.svg)](https://doi.org/10.5281/zenodo.22214012)

## 1. Abstract
The Recursive Reasoning Framework (RRF) is a domain-agnostic testing harness designed to move Large Language Model (LLM) agents away from "lucky guessing" and toward systematic discovery. Unlike standard evaluation harnesses that rely on large context windows or simple success/fail scalars, RRF enforces a **discrete cognitive layer** through a loop of hypothesis generation, deterministic backtesting, and cross-task memory persistence.

By decoupling environment logs from the prompt and utilizing an atomic JSON memory loop, RRF enables **Zero-Shot Knowledge Transfer**. An agent can solve a complex task using logical deductions inherited from past execution histories without requiring fine-tuning or prompt-bloat.

---

## 2. The Core Philosophy
* **Token Efficiency**: Instead of feeding entire execution histories into the LLM context, RRF provides a local API. The agent only processes the current state, using deterministic Python scripts to query history locally.
* **Deterministic Rigor**: Agents are restricted from testing hypotheses in the live environment if those hypotheses can be falsified by existing logs.
* **Mechanism Elucidation**: The harness mandates that agents state an explicit *predicted outcome* before acting, turning every action into a formal scientific experiment.
* **Recursive Meta-Learning**: Discovered knowledge is stored in a structured schema, allowing the framework to serve as a meta-learning complex for systematic exploration across varied domains.

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

To verify that the dynamic plugin system requires **no core edits**, a single file (`plugins/sokoban.py`) was introduced for Microban Level 1 (David Skinner, 2000).

* **Emergence of Optimal Pathing**: The system prompt (`SKILLS.template.md`) never explicitly mentions Breadth-First Search. Because exhaustive search is required to test spatial hypotheses reliably, the LLM autonomously inferred the need for a BFS algorithm. It wrote the necessary Python code locally, resulting in a 33-move, BFS-optimal path (`logs/microban_1_agent.log`). Every `POST /step` API call included a `predicted_outcome` that was successfully validated against the environment log prior to execution.
* **Rule Generalization and Cross-Level Transfer**: After solving Level 1, the agent recorded seven "high" confidence rules to `memory/global_memory.json` (e.g., walls block movement, win condition `boxes⊆goals`). When exposed to a new level (`microban_2`), the agent loaded this global memory, verified the generic rules with a single action, and inherited the physical mechanics zero-shot. Subsequent levels became easier because only the spatial layout required resolution, eliminating the need for physics rediscovery.

| Task | Boxes | Steps | Duration | Type |
| --- | --- | --- | --- | --- |
| `microban_1` | 2 | 33 | 63s | Exploratory (mechanics discovery) |
| `microban_2` | 3 | 16 | 102s | Inherited (physics reused) |

### Case Study 3: The Eleusis Variant (Contradiction Handling & Multi-Domain Filtering)

To validate non-stationary rule handling and cross-domain memory isolation, the **Eleusis (Inductive Sequence)** environment was introduced (`plugins/eleusis_env.py`). The hidden rule required an alternating number parity that inverted every 5 successes.

* **Contradiction Handling**: When a previously confirmed rule failed due to the 5-step phase inversion, the agent did not delete its history. Instead, it utilized a segmented backtest via Python to identify the change-point, downgraded the prior rule's confidence from `high` to `medium`, and recorded a new conditional phase-shift rule. 
* **Multi-Domain Filtering**: Following the Eleusis execution, the agent was tasked with a Sokoban level. The framework successfully isolated the memory schemas using the `domain` tag. The agent correctly ignored the parity rules and seamlessly initialized a spatial search algorithm, proving the persistent memory does not suffer from cross-domain logic interference.

### Case Study 4: Tic-Tac-Toe (Minimal Viable Exemplar)

The Tic-Tac-Toe plugin (`plugins/tictactoe_env.py`) validates the framework's ability to scale *down* to minimal complexity.

* **Optimal Choice of Algorithm**: Since the state space is highly constrained, the LLM extracted the deterministic rules directly from the execution history without use of a Breadth-First Search approach, proving the engine adapts its tool-use to task complexity.
* **Zero-Shot Inheritance**: A 62% reduction in execution time for the second task (`tictactoe_2` completed in 31 seconds versus 81 seconds for `tictactoe_1`) confirms the memory loop allows the agent to inherit high-confidence rules without a redundant exploration phase.

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

# Eleusis cross-task transfer (demonstrates phase-shift memory retention):
python3 -m core.harness --env eleusis_env --tasks eleusis_1 eleusis_2
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
* **Research**: Multi-domain filtering and non-stationary environment exploration.

### Stochastic Discovery & AI/Human Equivalence

**Stochastic rule inference is fully tractable** provided the underlying mechanism is discoverable. For example, in a 6-sided die plugin (`plugins/die_env.py`), the `step()` function returns a sampled integer along with a `variance_proxy`. This enables the LLM to run `scipy.stats.chisquare` against the execution logs, fit a uniform distribution, and write a "medium" or "low" confidence rule to the global memory.

Because predicting a single roll cannot exceed a 1/6 probability, the agent relies on contradiction-handling protocols to downgrade confidence rather than asserting absolute truth. The explicit backtest loop ensures the AI provides an auditable, reproducible history of its distribution analysis.

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

## 7. Architectural Boundaries and Epistemological Limits

To properly position RRF within the broader artificial intelligence landscape, it is essential to define both its capabilities and boundaries.

### The Verifiability Bottleneck
At its core, RRF requires an objective, computationally testable verifier. If an outcome cannot be programmatically validated or falsified, the system breaks down. This fundamental requirement is shared with training paradigms like Reinforcement Learning with Verifiable Rewards (RLVR); truth must be computationally testable. 

### RRF Limits vs. RLVR
While operating on similar epistemological constraints, RRF is significantly more bounded by design:
* **Inference-Time Orchestration:** RRF achieves verification strictly through an external JSON memory loop and prompt orchestration. It never updates the model's neural weights.
* **Algorithmic Recall:** The capability recursion is explicitly bounded by the LLM's pre-existing knowledge of algorithms. The framework acts as a multiplier to deploy known computational tools (e.g., Breadth-First Search, mathematical libraries). It will not spontaneously invent novel algorithms outside its pre-training data.

### The State-Space Ceiling (The Chess Boundary)
Directly scaling to complex, unconstrained environments such as standard Chess violates the framework’s core philosophy:
* **Action Space Inflation:** Injecting 30+ legal moves into the context window at every step degrades token efficiency.
* **Unfalsifiable State Spaces:** With approximately 10^44 possible positions, standard whole-log backtesting becomes impossible. An agent cannot falsify a complex positional heuristic based on a localized log of a few games.
* **Stochastic Adversaries:** A hidden, non-deterministic opponent policy removes the exact falsifiability requirement to log a rule with high confidence.

A claim is valid only when it is **knowable** (the model understands the test), **testable** (the environment exposes necessary variables), **tractable** (the log is small enough for complete replay), and **falsifiable**.

---

## 8. Application in the Natural Sciences

Despite strict boundaries in unbounded board games, computational and experimental natural sciences are the most suitable domains for RRF. Physical laws and biological networks are defined by underlying quantitative rules and bounded state spaces, aligning perfectly with the digitized scientific method.

### Algorithmic Offloading
Language models natively struggle with complex raw computation. RRF resolves this by acting as a bridge between symbolic hypothesis generation and rigorous quantitative verification. If a problem requires combinatorial search, statistical distributions, or complex algebra, the agent autonomously writes and executes Python code (utilizing libraries such as `scipy`, `numpy`, or `sympy`). The deterministic output informs the conclusion, effectively eliminating hallucinated mathematical logic.

### Probabilistic Verification
Transitioning from discrete puzzle environments to biological or chemical data introduces a necessary shift from rigid boolean truth to statistical significance.
* **Statistical Falsification:** In domains containing natural noise, the framework evaluates rules based on statistical confidence (e.g., p-values, residual errors) rather than absolute contradictions. A rule may be validated but tagged with a "Medium" confidence level based on variance.
* **Bioinformatics Integration:** Within Gene Regulatory Network Discovery, biological data is inherently probabilistic. An agent can navigate raw gene expression datasets by querying specific experimental conditions. It proposes a biological mechanism, predicts the statistical signature expected, and writes a script to perform variance or correlation analysis. 
* **Zero-Shot Transfer in Biology:** Once a regulatory relationship survives rigorous statistical backtesting, it is recorded in the global memory schema. When assigned a new task within the same domain, such as analyzing a mutant cell line with a specific gene deletion, the agent inherits the previously verified network map to deduce cascading effects without redundant data analysis.

---

### License

* This project is licensed under the MIT License.

### Citation

* Friedman, R. (2026) *Recursive Reasoning Framework* (v1.2). Zenodo. https://doi.org/10.5281/zenodo.22214012

### Acknowledgments

* The conceptual development of the methodology and codebase benefited from discussions and iterative refinement with AI language models, Gemini 3.1 Pro (Google) and Muse Spark 1.2 (Meta).