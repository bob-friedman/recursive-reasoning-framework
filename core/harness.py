#!/usr/bin/env python3
"""
harness.py

Main execution engine for the Recursive Reasoning Framework (RRF).
Handles HTTP server creation, process isolation, atomic writes, and agent execution.
"""

import argparse
import datetime
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import inspect
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure standard import paths work when executed directly
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
from core.environment import BaseEnvironment

# Initialize config
secrets_path = Path.home() / ".env"
load_dotenv(dotenv_path=secrets_path)

# Directories
LOG_DIR = BASE_DIR / "logs"
MEMORY_DIR = BASE_DIR / "memory"
SKILLS_TEMPLATE = BASE_DIR / "SKILLS.template.md"
PROVIDER_LOG = LOG_DIR / "provider.log"
HARNESS_LOG = LOG_DIR / "harness_stdout.log"

SENSITIVE_KEY_PATTERNS = [
    r".*_API_KEY$", r".*_SECRET.*", r".*_TOKEN.*", r"SECRET_.*",
    r"^OPENAI_.*", r"^ANTHROPIC_.*", r"^DEEPSEEK_.*", r"^GITHUB_.*",
    r"^AWS_.*", r"^AZURE_.*", r"^GCP_.*", r".*_PASSWORD$", r".*_PRIVATE_KEY$"
]

def sanitize_task_id(task_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", task_id)
    return safe if safe else "task"

def create_safe_env() -> Dict[str, str]:
    safe_env = {}
    for key, value in os.environ.items():
        if not any(re.match(pattern, key) for pattern in SENSITIVE_KEY_PATTERNS):
            safe_env[key] = value
    return safe_env

# ==============================================================================
# Atomic Write & Schema Validation
# ==============================================================================
MEMORY_SCHEMA = {
    "type": "object",
    "required": ["schema", "rules"],
    "properties": {
        "schema": {"type": "string"},
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["rule", "confidence", "domain"],
                "properties": {
                    "rule": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "domain": {"type": "string"},
                },
            },
        },
    },
}

# Executable skill library, separate from declarative memory rules.
# RRF agents already invent BFS/stats scripts ad-hoc (Sokoban, die_env);
# banking them here stops rediscovery. Ported from continual-harness
# skill-store concept, trimmed to RRF's atomic-JSON style.
SKILLS_SCHEMA = {
    "type": "object",
    "required": ["schema", "skills"],
    "properties": {
        "schema": {"type": "string"},
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "description"],
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "code": {"type": "string"},
                    "effectiveness": {"type": "string", "enum": ["high", "medium", "low"]},
                    "domain": {"type": "string"},
                },
            },
        },
    },
}

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}

def _validate_value(value: Any, schema: Dict) -> bool:
    expected_type = schema.get("type")
    if expected_type == "string": return isinstance(value, str)
    elif expected_type == "array":
        if not isinstance(value, list): return False
        items_schema = schema.get("items", {})
        return all(_validate_value(item, items_schema) for item in value)
    elif expected_type == "object":
        if not isinstance(value, dict): return False
        for prop, prop_schema in schema.get("properties", {}).items():
            if prop in schema.get("required", []) and prop not in value: return False
            if prop in value and not _validate_value(value[prop], prop_schema): return False
        return True
    elif expected_type == "boolean": return isinstance(value, bool)
    elif expected_type == "number": return isinstance(value, (int, float))
    return True

def validate_json_schema(data: Any, schema: Dict) -> bool:
    try: return _validate_value(data, schema)
    except Exception: return False

def atomic_write_json(path: Path, data: Any, validate_schema: Optional[Dict] = None) -> None:
    if validate_schema and not validate_json_schema(data, validate_schema):
        raise ValueError(f"Data does not match required schema for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as tmp:
        tmp.write(json_dumps(data, indent=2, ensure_ascii=False))
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)

def atomic_read_json(path: Path, default: Any = None, validate_schema: Optional[Dict] = None) -> Any:
    if not path.exists(): return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        if validate_schema and not validate_json_schema(data, validate_schema): return default
        return data
    except (json.JSONDecodeError, OSError): return default

def find_free_port(start_port: int = 8080) -> int:
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0: return port
        port += 1
    raise RuntimeError("No free ports available.")

# ==============================================================================
# Logging Utilities
# ==============================================================================
def _setup_harness_logging():
    LOG_DIR.mkdir(exist_ok=True)
    MEMORY_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("harness")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(HARNESS_LOG, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        pfh = logging.FileHandler(PROVIDER_LOG, encoding="utf-8")
        pfh.setFormatter(fmt)
        logger.addHandler(pfh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

HARNESS_LOGGER = _setup_harness_logging()

# ==============================================================================
# Truncation & Session Handling
# ==============================================================================
MAX_CONTEXT_CHARS = 50000

def truncate_observation(obs: Any, max_chars: int = MAX_CONTEXT_CHARS) -> Any:
    json_str = json_dumps(obs, ensure_ascii=False)
    if len(json_str) <= max_chars: return obs
    if isinstance(obs, dict):
        truncated, remaining = {}, max_chars - 100
        for key, value in obs.items():
            val_str = json_dumps(value, ensure_ascii=False)
            if len(val_str) <= remaining:
                truncated[key] = value
                remaining -= len(val_str)
            else:
                truncated[key] = f"<truncated: {len(val_str)} chars>"
                break
        truncated["_truncated"] = True
        return truncated
    return {"_truncated": True, "summary": json_str[:max_chars]}

class TaskSession:
    MAX_OUTCOME_LOG_SIZE = 10000
    def __init__(self, env: BaseEnvironment, task_id: str, max_actions: int):
        self.env = env
        self.task_id = task_id
        self.safe_task_id = sanitize_task_id(task_id)
        self.max_actions = max_actions
        self.step_count = 0
        self.done = False
        self.is_win = False
        self.obs = None
        self.outcome_log = []
        self._lock = threading.Lock()
        self.log_path = LOG_DIR / f"{self.safe_task_id}_env.log"
        self.log_path.write_text("")
        # Structured JSONL trajectory (machine-readable twin of env.log).
        # env.log stays human-readable for backtest scripts; trajectory.jsonl
        # carries pre/post obs + predicted_outcome for failure-pattern analysis.
        self.trajectory_path = LOG_DIR / f"{self.safe_task_id}_trajectory.jsonl"
        self.trajectory_path.write_text("")
        self.obs = self.env.start(task_id)
        self.log_frame()

    def log_frame(self):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"--- step {self.step_count} ---\n")
            f.write(json_dumps(self.obs, ensure_ascii=False) + "\n")

    def log_trajectory(self, entry: Dict[str, Any]):
        with open(self.trajectory_path, "a", encoding="utf-8") as f:
            f.write(json_dumps(entry, ensure_ascii=False) + "\n")

    def step(self, action_payload: Dict[str, Any]):
        with self._lock:
            if self.done: raise ValueError("Task session is already completed.")
            # --- Science gate (kept intact): predicted_outcome is mandatory ---
            pred = action_payload.get("predicted_outcome", "")
            if not isinstance(pred, str) or not pred.strip():
                raise ValueError(
                    "Missing 'predicted_outcome'. Every /step must state an explicit "
                    "prediction, e.g. {\"action\": \"PRESS_B\", \"predicted_outcome\": "
                    "\"color stays White, history=[PRESS_B]\"}. Test against the log first."
                )
            # --- Action validation via plugin contract (if implemented) ---
            try:
                err = self.env.validate_action(action_payload)
            except Exception:
                err = None
            if err:
                raise ValueError(err)
            pre_obs = self.obs
            self.obs, env_done, self.is_win = self.env.step(action_payload)
            self.step_count += 1
            self.log_frame()
            self.done = env_done or self.is_win or (self.step_count >= self.max_actions)
            if len(self.outcome_log) >= self.MAX_OUTCOME_LOG_SIZE:
                self.outcome_log = self.outcome_log[-self.MAX_OUTCOME_LOG_SIZE // 2:]
            self.outcome_log.append({
                "step": self.step_count,
                "payload": action_payload,
                "actual_state": json_dumps(self.obs, ensure_ascii=False),
            })
            # Structured trajectory entry: prediction vs actual, both states.
            # Never fails the step on logging errors.
            try:
                self.log_trajectory({
                    "step": self.step_count,
                    "task_id": self.task_id,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "action": str(action_payload.get("action", "")),
                    "predicted_outcome": pred,
                    "pre_obs": pre_obs,
                    "post_obs": self.obs,
                    "done": self.done,
                    "is_win": self.is_win,
                })
            except Exception:
                HARNESS_LOGGER.warning("trajectory log failed at step %d", self.step_count, exc_info=True)
            return {"step": self.step_count, "done": self.done, "is_win": self.is_win, "obs": self.obs}

# ==============================================================================
# Server & Request Extraction
# ==============================================================================
class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

def extract_json_from_markdown(text: str) -> Optional[Dict]:
    if not text: return None
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text.strip(), re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except json.JSONDecodeError: pass
    if text.strip().startswith("{"):
        try: return json.loads(text.strip())
        except json.JSONDecodeError: pass
    return None

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime): return obj.isoformat()
        if isinstance(obj, Path): return str(obj)
        if isinstance(obj, set): return list(obj)
        if hasattr(obj, "__dict__"): return obj.__dict__
        return str(obj)

def json_dumps(obj: Any, **kwargs) -> str:
    return json.dumps(obj, cls=JSONEncoder, **kwargs)

def create_api_handler(session: TaskSession):
    class APIHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args): pass
        def send_json(self, status_code: int, payload: Dict):
            self.send_response(status_code)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json_dumps(payload, ensure_ascii=False).encode("utf-8"))

        def do_GET(self):
            if self.path == "/state":
                with session._lock:
                    self.send_json(200, {
                        "task_id": session.task_id, "step": session.step_count,
                        "done": session.done, "is_win": session.is_win,
                        "obs": truncate_observation(session.obs),
                    })
            else: self.send_json(404, {"error": "Not Found"})

        def do_POST(self):
            if self.path == "/step":
                post_data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                try:
                    try: req = json.loads(post_data)
                    except json.JSONDecodeError:
                        req = extract_json_from_markdown(post_data.decode("utf-8", errors="replace"))
                        if req is None:
                            self.send_json(400, {"error": "Invalid JSON format"})
                            return
                    resp = session.step(req)
                    self.send_json(200, resp)
                except ValueError as ve: self.send_json(400, {"error": str(ve)})
                except Exception as e:
                    traceback.print_exc()
                    self.send_json(500, {"error": f"Internal Error: {str(e)}"})
            else: self.send_json(404, {"error": "Not Found"})
    return APIHandler

# ==============================================================================
# Agent Execution
# ==============================================================================
def ensure_skills_store() -> Path:
    skills_path = MEMORY_DIR / "skills.json"
    existing = atomic_read_json(skills_path, default=None, validate_schema=SKILLS_SCHEMA)
    if existing is None:
        existing = {
            "schema": "Reusable executable strategies. Schema: [{'name': '...', 'description': 'when to use', 'code': 'inline python reading args, setting result', 'effectiveness': 'high|medium|low', 'domain': '...'}]",
            "skills": [],
        }
        atomic_write_json(skills_path, existing, validate_schema=SKILLS_SCHEMA)
    return skills_path


def bootstrap_from(source: str) -> Dict[str, Any]:
    """Import memory/skills from a prior run directory (continual-harness style).

    Accepts a directory containing global_memory.json and/or skills.json,
    or a direct path to one of those files. Merges by content:
    - memory rules deduped by rule text, keeping higher confidence
    - skills deduped by name, keeping the incoming entry on conflict
    Returns summary counts.
    """
    src = Path(source)
    mem_src, skills_src = None, None
    if src.is_dir():
        cand_mem = src / "global_memory.json"
        if cand_mem.exists():
            mem_src = cand_mem
        # also accept continual-harness naming
        if mem_src is None and (src / "memory.json").exists():
            mem_src = src / "memory.json"
        cand_sk = src / "skills.json"
        if cand_sk.exists():
            skills_src = cand_sk
    elif src.is_file():
        low = src.name.lower()
        if "skill" in low:
            skills_src = src
        else:
            mem_src = src
    else:
        raise ValueError(f"--bootstrap-from path does not exist: {source}")

    summary: Dict[str, Any] = {"memory_imported": 0, "skills_imported": 0}
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    if mem_src is not None:
        incoming = atomic_read_json(mem_src, default=None)
        if incoming is not None:
            # normalize continual-harness {entries:{...}} -> RRF {rules:[...]}
            if "rules" not in incoming and "entries" in incoming:
                incoming = {
                    "schema": incoming.get("schema", ""),
                    "rules": [
                        {"rule": f"{e.get('title','')}: {e.get('content','')}".strip(": "),
                         "confidence": "medium" if e.get("importance", 3) >= 3 else "low",
                         "domain": str(e.get("path", "general")).split("/")[0]}
                        for e in incoming["entries"].values()
                    ],
                }
            current = atomic_read_json(MEMORY_DIR / "global_memory.json", default=None,
                                       validate_schema=MEMORY_SCHEMA)
            if current is None:
                current = {"schema": incoming.get("schema", "rules"), "rules": []}
            by_rule = {r["rule"]: r for r in current.get("rules", [])}
            added = 0
            for r in incoming.get("rules", []):
                if not isinstance(r, dict) or "rule" not in r:
                    continue
                rule_text = str(r["rule"])
                conf = str(r.get("confidence", "low")).lower()
                if conf not in CONFIDENCE_RANK:
                    conf = "low"
                dom = str(r.get("domain", "general"))
                prev = by_rule.get(rule_text)
                if prev is None:
                    by_rule[rule_text] = {"rule": rule_text, "confidence": conf, "domain": dom}
                    added += 1
                elif CONFIDENCE_RANK[conf] > CONFIDENCE_RANK.get(str(prev.get("confidence", "low")).lower(), 0):
                    by_rule[rule_text] = {"rule": rule_text, "confidence": conf, "domain": dom}
            current["rules"] = sorted(by_rule.values(), key=lambda x: x["rule"])
            atomic_write_json(MEMORY_DIR / "global_memory.json", current, validate_schema=MEMORY_SCHEMA)
            summary["memory_imported"] = added

    if skills_src is not None:
        incoming = atomic_read_json(skills_src, default=None)
        if incoming is not None:
            if "skills" not in incoming and "entries" in incoming:
                incoming = {"schema": "", "skills": [
                    {"name": e.get("name", eid), "description": e.get("description", ""),
                     "code": e.get("code", ""), "effectiveness": e.get("effectiveness", "medium"),
                     "domain": str(e.get("path", "general")).split("/")[0]}
                    for eid, e in incoming["entries"].items()]}
            skills_path = ensure_skills_store()
            current = atomic_read_json(skills_path, default=None, validate_schema=SKILLS_SCHEMA)
            by_name = {s["name"]: s for s in current.get("skills", [])}
            added = 0
            for s in incoming.get("skills", []):
                if not isinstance(s, dict) or "name" not in s:
                    continue
                name = str(s["name"])
                if name not in by_name:
                    added += 1
                by_name[name] = {
                    "name": name,
                    "description": str(s.get("description", "")),
                    "code": str(s.get("code", "")),
                    "effectiveness": str(s.get("effectiveness", "medium")).lower()
                    if str(s.get("effectiveness", "medium")).lower() in ("high", "medium", "low") else "medium",
                    "domain": str(s.get("domain", "general")),
                }
            current["skills"] = sorted(by_name.values(), key=lambda x: x["name"])
            atomic_write_json(skills_path, current, validate_schema=SKILLS_SCHEMA)
            summary["skills_imported"] = added

    HARNESS_LOGGER.info("bootstrap-from %s: %s", source, summary)
    return summary


def run_task_agent(session: TaskSession, port: int, timeout: int) -> float:
    global_memory_path = MEMORY_DIR / "global_memory.json"
    initial_memory = atomic_read_json(global_memory_path, default=None, validate_schema=MEMORY_SCHEMA)
    if initial_memory is None:
        initial_memory = {
            "schema": "List of confirmed rules. Schema: [{'rule': '...', 'confidence': 'high|medium|low', 'domain': '...'}]",
            "rules": [],
        }
        atomic_write_json(global_memory_path, initial_memory, validate_schema=MEMORY_SCHEMA)

    skills_path = ensure_skills_store()

    skills_text = SKILLS_TEMPLATE.read_text(encoding="utf-8")
    valid_actions_str = ", ".join(session.env.get_valid_actions())
    skills_text = skills_text.replace("{BASE_URL}", f"http://127.0.0.1:{port}").replace("{VALID_ACTIONS}", valid_actions_str)

    (LOG_DIR / f"{session.safe_task_id}_skills.md").write_text(skills_text, encoding="utf-8")

    autonomy_addendum = (
        f"\n--- AUTONOMY ADDENDUM (continual-harness lite, science gate intact) ---\n"
        f"Structured trajectory (machine-readable, one JSON per step with "
        f"action/predicted_outcome/pre_obs/post_obs): {session.trajectory_path}\n"
        f"Use it for failure-pattern checks (loops = same obs 2 steps apart, "
        f"blocked-move = pre_obs == post_obs after movement action) before acting.\n"
        f"Reusable skill library: {skills_path}\n"
        f"- Before writing throwaway analysis code, load skills.json and reuse a matching "
        f"skill (same domain, effectiveness high first).\n"
        f"- If you write a generally useful script (BFS, changepoint scan, chi-square), "
        f"save it to skills.json as {{name, description, code, effectiveness, domain}} "
        f"with code as inline Python reading `args` and setting `result`.\n"
        f"- If you call the equivalent of run_code >=3 times without checking skills, "
        f"stop and codify instead of duplicating.\n"
        f"Memory hygiene: domain = plugin canonical domain (e.g. `{session.env.get_domain()}`), "
        f"never the task id; dedupe rules by text keeping higher confidence.\n"
        f"The /step gate is enforced: missing or empty predicted_outcome returns HTTP 400 "
        f"and does not advance the environment. This is non-negotiable.\n"
        f"--------------------------------------------------------------------\n"
    )

    prompt = (
        f"The agent is an AI solving the task: {session.task_id}.\n"
        f"The environment is hosted on a local server at http://127.0.0.1:{port}\n\n"
        f"API Endpoints:\n"
        f"- GET http://127.0.0.1:{port}/state  -> View current state\n"
        f"- POST http://127.0.0.1:{port}/step  -> Take action. Payload strictly JSON: "
        f'{{"action": "<ACTION_NAME>", "predicted_outcome": "<prediction text>"}}\n\n'
        f"Valid Actions: {valid_actions_str}.\n\n"
        f"The server automatically logs every frame encountered to: {session.log_path}\n"
        f"Structured trajectory JSONL: {session.trajectory_path}\n"
        f"The structured cross-task memory file (JSON) is located at: {global_memory_path}\n"
        f"Reusable skills file (JSON) is located at: {skills_path}\n\n"
        f"--- WORKFLOW AND SKILLS ---\n{skills_text}\n---------------------------\n"
        f"{autonomy_addendum}\n"
        f"Instructions:\n"
        f"1. The agent must strictly adhere to the workflow detailed above.\n"
        f"2. Avoid guessing. Before calling the /step API, write and execute Python code to test "
        f"the hypothesis against the log file.\n"
        f"3. When the API returns 'done': true (or 'is_win': true), the task is complete. "
        f"Update the global memory JSON with confirmed rules, then terminate."
    )

    print(f"Spawning opencode agent for {session.task_id}...")
    start_time, proc = time.time(), None
    try:
        proc = subprocess.Popen(
            ["opencode", "run", prompt], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=create_safe_env(), start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try: stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                stdout, stderr = proc.communicate()
            returncode = -1

        (LOG_DIR / f"{session.safe_task_id}_agent.log").write_text(
            f"--- RETURNCODE: {returncode} DURATION: {time.time() - start_time:.2f}s ---\n"
            f"--- STDOUT ---\n{stdout}\n--- STDERR ---\n{stderr}\n", encoding="utf-8"
        )
    except Exception as e:
        if proc and proc.poll() is None:
            try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception: pass
        (LOG_DIR / f"{session.safe_task_id}_agent.log").write_text(
            f"--- EXCEPTION after {time.time() - start_time:.2f}s ---\n{traceback.format_exc()}\n", encoding="utf-8"
        )

    return time.time() - start_time

def sanitize_env_name(env_name: str) -> str:
    """Allow only safe Python module names to prevent path traversal."""
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", env_name):
        raise ValueError(f"Invalid env name '{env_name}': must match ^[a-zA-Z_][a-zA-Z0-9_]*$")
    return env_name


def list_available_plugins() -> Dict[str, str]:
    """
    Scan plugins/ for importable modules. Returns {module_name: description}.
    Does not require hard-coded registration.
    """
    plugins_dir = BASE_DIR / "plugins"
    available: Dict[str, str] = {}
    if not plugins_dir.exists():
        return available
    for p in plugins_dir.glob("*.py"):
        if p.name.startswith("_"):
            continue
        mod_name = p.stem
        # Try lightweight docstring extraction without importing
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            # first class docstring heuristic
            m = re.search(r'class\s+\w+\(BaseEnvironment\)\s*:\s*"""(.*?)"""', text, re.DOTALL)
            if not m:
                m = re.search(r'class\s+\w+\(BaseEnvironment\)\s*:\s*\'\'\'(.*?)\'\'\'', text, re.DOTALL)
            desc = m.group(1).strip().splitlines()[0][:120] if m else ""
        except Exception:
            desc = ""
        available[mod_name] = desc
    return available


def get_plugin_environment(env_name: str) -> BaseEnvironment:
    """
    Truly dynamic plugin loader.
    - Sanitizes env_name to prevent traversal
    - Uses importlib to load plugins.{env_name}
    - Finds first concrete subclass of BaseEnvironment in the module
    - Raises with helpful list of available plugins on failure
    """
    env_name = sanitize_env_name(env_name)
    try:
        module = importlib.import_module(f"plugins.{env_name}")
    except ModuleNotFoundError as e:
        available = list_available_plugins()
        hint = f" Available: {sorted(available.keys())}" if available else ""
        raise ValueError(f"Unknown environment plugin '{env_name}'.{hint}") from e

    # Find concrete subclasses
    candidates = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseEnvironment) and obj is not BaseEnvironment:
            # ensure class was defined in this module, not imported
            if obj.__module__ == module.__name__:
                candidates.append(obj)

    if not candidates:
        raise ValueError(f"Plugin '{env_name}' loaded but no BaseEnvironment subclass found in plugins/{env_name}.py")
    if len(candidates) > 1:
        HARNESS_LOGGER.warning(f"Plugin '{env_name}' has multiple env classes {candidates}, using {candidates[0].__name__}")
    return candidates[0]()


def main():
    parser = argparse.ArgumentParser(description="Recursive Reasoning Framework Harness")
    parser.add_argument("--env", default="state_machine", help="Environment plugin to load (e.g., state_machine)")
    parser.add_argument("--tasks", nargs="*", default=["task_1_blue", "task_2_green"], help="List of task IDs")
    parser.add_argument("--results", default="results.json", help="Destination summary file")
    parser.add_argument("--max-actions", type=int, default=100, help="Max actions per task")
    parser.add_argument("--timeout", type=int, default=300, help="Agent timeout per task in seconds")
    parser.add_argument("--port", type=int, default=None, help="Force specific server port")
    parser.add_argument("--bootstrap-from", default=None, help="Import memory/skills from prior run dir (contains global_memory.json and/or skills.json) or a single JSON file")
    parser.add_argument("--list-envs", action="store_true", help="List available plugins and exit")
    args = parser.parse_args()

    if args.list_envs:
        available = list_available_plugins()
        if not available:
            print("No plugins found in plugins/")
        else:
            print("Available plugins:")
            for name, desc in sorted(available.items()):
                print(f"  - {name}: {desc}")
            # Try to show richer metadata by instantiating
            for name in sorted(available.keys()):
                try:
                    env = get_plugin_environment(name)
                    print(f"    -> {name}: domain={env.get_domain()}, actions={env.get_valid_actions()}, desc={env.get_description()[:80]}")
                except Exception as e:
                    print(f"    -> {name}: load error: {e}")
        sys.exit(0)

    LOG_DIR.mkdir(exist_ok=True)
    MEMORY_DIR.mkdir(exist_ok=True)
    ensure_skills_store()

    if args.bootstrap_from:
        try:
            summary = bootstrap_from(args.bootstrap_from)
            print(f"Bootstrap import: {summary} from {args.bootstrap_from}")
        except Exception as e:
            print(f"ERROR in --bootstrap-from: {e}", file=sys.stderr)
            sys.exit(3)

    if not SKILLS_TEMPLATE.exists():
        print(f"ERROR: {SKILLS_TEMPLATE} not found.", file=sys.stderr)
        sys.exit(1)

    try:
        env = get_plugin_environment(args.env)
    except ValueError as ve:
        print(f"ERROR: {ve}", file=sys.stderr)
        sys.exit(2)
    results = []

    for task_id in args.tasks:
        print(f"\nInitializing task session: {task_id}")
        port = args.port if args.port else find_free_port()
        session = TaskSession(env, task_id, args.max_actions)

        server = ReusableThreadingHTTPServer(("127.0.0.1", port), create_api_handler(session))
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        print(f"Background server listening on port {port}")
        try:
            duration = run_task_agent(session, port, args.timeout)
            results.append({
                "task_id": task_id, "solved": session.is_win, "actions_taken": session.step_count,
                "duration_sec": round(duration, 2),
                "trajectory": str(session.trajectory_path),
                "env_log": str(session.log_path),
                "outcome_log": session.outcome_log,
            })
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)

        print(f"Session result for {task_id}: Win={session.is_win}, Steps={session.step_count}")
        atomic_write_json(Path(args.results), {"results": results, "status": "partial"})

    atomic_write_json(Path(args.results), {"results": results, "status": "complete"})
    print(f"\nExecution complete. Summary generated at {args.results}")

if __name__ == "__main__":
    main()

