# Task Solving Skill

## Core Principle
Do not execute an action to test a hypothesis that can be verified against history. The agent must write and execute Python code to replay hypotheses against the recorded log before interacting with the API.
- A past state contradicting the hypothesis falsifies it.
- If the log cannot settle the hypothesis, only then take a real action.
- Every real action must carry a stated predicted outcome.

## Environment Architecture
The agent interacts with a local HTTP server at `{BASE_URL}`.
- **Observe:** Fetch current state via `GET {BASE_URL}/state`
- **Log:** The server automatically appends every state to a local environment log file (path provided in the prompt).
- **Act:** Take actions via `POST {BASE_URL}/step`.
- **Valid Actions:** `{VALID_ACTIONS}`

## Workflow
1. **Observe:** Execute a GET request for the current state and review the environment.
2. **Hypothesize:** Propose a specific, testable rule regarding mechanics.
3. **Memory Check:** Load `memory/global_memory.json`; **first read the `schema` field**, then if a confirmed rule (`confidence: high`) exists for this task and domain, verify it with ONE action before deriving new rules.
4. **Backtest:** Write and execute a Python script to test the hypothesis against all past states in the environment log.
   - *Falsified:* Discard and revise hypothesis immediately.
   - *Confirmed / Untestable from history:* Proceed to **Step 5**.
   - *Insufficient History:* If the log is empty or lacks relevant states, state this clearly before taking an exploratory action.
5. **Act:** Send a POST request to execute one real action. The explicit predicted outcome must be included in the JSON payload.
6. **Verify:** Read the new state. Compare the actual state to the prediction.
   - *Match:* Mark hypothesis verified.
   - *Miss:* Mark falsified and immediately revise. **You must explain why the prediction differed from the actual state before forming a new hypothesis.**
7. **Update Memory:** Update the structured JSON memory file (provided in the prompt) with newly confirmed rules. Remove falsified rules. **Strictly adhere to the `schema` field defined within the memory file** to prevent corrupting the self-recursive learning data. Ensure valid JSON format.
8. **Repeat:** Continue until the API returns `done: true` or `is_win: true`. Carry confirmed general rules forward into the global memory for subsequent tasks.

## Directives
- Write Python scripts to check logs before taking actions.
- Keep hypotheses narrow and testable.
- Discard falsified rules immediately.
- Avoid calling the POST endpoint without prior log verification.
- Trust high-confidence memory rules first; verify with minimal actions before re-deriving.
- When outputting JSON for the `/step` API, wrap in markdown code blocks (```json ... ```) if desired; the harness will extract it.

