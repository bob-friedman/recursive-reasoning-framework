# Task Solving Skill — Eleusis Variant (Additive, Non-Breaking)
# Preserves SKILLS.template.md core; extends Step 4 for non-stationary phase shift.
# Use via: core/harness.py will load SKILLS.template.md by default;
# for eleusis tasks, operator may symlink or pass this file. Default deterministic
# behavior for state_machine/sokoban/counter/die remains untouched.

## Core Principle (unchanged)
Do not execute an action to test a hypothesis that can be verified against history. The agent must write and execute Python code to replay hypotheses against the recorded log before interacting with the API.
- A past state contradicting the hypothesis falsifies it.
- If the log cannot settle the hypothesis, only then take a real action.
- Every real action must carry a stated predicted outcome.

## Environment Architecture
The agent interacts with a local HTTP server at `{BASE_URL}`.
- **Observe:** Fetch current state via `GET {BASE_URL}/state`
- **Log:** The server automatically appends every state to a local environment log file (path provided in the prompt).
- **Act:** Take actions via `POST {BASE_URL}/step`.
- **Valid Actions:** `{VALID_ACTIONS}` — for Eleusis: `PLAY_1`..`PLAY_10` (or `{"value": int, "action": "PLAY_X"}`)

## Workflow
1. **Observe:** Execute a GET request for the current state and review the environment.
2. **Hypothesize:** Propose a specific, testable rule regarding mechanics.
3. **Memory Check:** Load `memory/global_memory.json`; **first read the `schema` field**, then if a confirmed rule (`confidence: high`) exists for this domain (`inductive_logic`), verify it with ONE action before deriving new rules.
4. **Backtest (Eleusis-extended):** Write and execute a Python script to test the hypothesis against all past states in the environment log.
    - *Stationary check:* Test hypothesis against entire log — if falsified on every segment, discard.
    - *Non-stationary check (additive):* If a `high` confidence rule that previously held now fails, DO NOT discard entire history. Instead:
      a) Split the log at candidate change-points (`length % 5 == 0` and brute-force `k=3..6`). For each `k`, score `H1` on `length < k` and `H2` (inverted parity) on `length >= k`.
      b) The `k` with maximal score is the hypothesized phase inversion. Validate `phase-shift model: (val>prev if (length%2==0 xor phase==2) else val<prev)` achieves 100% on segmented log.
      c) Downgrade prior rule `high→medium→low` (README Future #2) and store new conditional rule: `{"rule": "phase 1: even→greater odd→smaller; phase 2 inverted; flips every 5 successes", "confidence": "high", "domain": "inductive_logic"}` with phase condition.
    - *Falsified:* Discard and revise hypothesis immediately (including explaining which segment failed).
    - *Confirmed / Untestable from history:* Proceed to **Step 5**.
    - *Insufficient History:* If the log is empty or lacks relevant states, state this clearly before taking an exploratory action.
5. **Act:** Send a POST request to execute one real action. The explicit predicted outcome must be included in the JSON payload (`{"action": "PLAY_X", "value": X, "predicted_outcome": "..."}`).
6. **Verify:** Read the new state. Compare the actual state to the prediction.
    - *Match:* Mark hypothesis verified.
    - *Miss:* Mark falsified and immediately revise. **You must explain why the prediction differed from the actual state (e.g., phase flipped) before forming a new hypothesis.**
7. **Update Memory:** Update the structured JSON memory file (provided in the prompt) with newly confirmed rules. Remove or downgrade falsified rules. **Strictly adhere to the `schema` field** (`high|medium|low`). For Eleusis, never store singleton `greater/smaller` without phase condition after step 5. Before writing, load the current file, dedupe `rules` by `rule` text keeping the higher-confidence copy, and set each `domain` to the current plugin's canonical domain (from the plugin source) — not the task id.
8. **Targeted Search (Eleusis terminal):** Once phase model is confirmed, do NOT guess for `is_win: length>=10 && even`. Write Python BFS/DFS over `1..10` using the phase-aware predicate to find the minimal valid even terminal sequence, then execute it stepwise with predictions.
9. **Repeat:** Continue until the API returns `done: true` or `is_win: true`. Carry confirmed general rules forward into global memory for subsequent tasks.

## Directives (unchanged + one additive)
- Write Python scripts to check logs before taking actions.
- Keep hypotheses narrow and testable.
- Discard or downgrade falsified rules immediately.
- Avoid calling the POST endpoint without prior log verification.
- Trust high-confidence memory rules first; verify with minimal actions before re-deriving.
- When outputting JSON for the `/step` API, wrap in markdown code blocks (```json ... ```) if desired; the harness will extract it.
- For non-stationary logs, always test phase-inversion hypothesis before declaring environment random.
