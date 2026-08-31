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
        self.obs = self.env.start(task_id)
        self.log_frame()

    def log_frame(self):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"--- step {self.step_count} ---\n")
            f.write(json_dumps(self.obs, ensure_ascii=False) + "\n")

    def step(self, action_payload: Dict[str, Any]):
        with self._lock:
            if self.done: raise ValueError("Task session is already completed.")
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
def run_task_agent(session: TaskSession, port: int, timeout: int) -> float:
    global_memory_path = MEMORY_DIR / "global_memory.json"
    initial_memory = atomic_read_json(global_memory_path, default=None, validate_schema=MEMORY_SCHEMA)
    if initial_memory is None:
        initial_memory = {
            "schema": "List of confirmed rules. Schema: [{'rule': '...', 'confidence': 'high|medium|low', 'domain': '...'}]",
            "rules": [],
        }
        atomic_write_json(global_memory_path, initial_memory, validate_schema=MEMORY_SCHEMA)

    skills_text = SKILLS_TEMPLATE.read_text(encoding="utf-8")
    valid_actions_str = ", ".join(session.env.get_valid_actions())
    skills_text = skills_text.replace("{BASE_URL}", f"http://127.0.0.1:{port}").replace("{VALID_ACTIONS}", valid_actions_str)
    
    (LOG_DIR / f"{session.safe_task_id}_skills.md").write_text(skills_text, encoding="utf-8")

    prompt = (
        f"The agent is an AI solving the task: {session.task_id}.\n"
        f"The environment is hosted on a local server at http://127.0.0.1:{port}\n\n"
        f"API Endpoints:\n"
        f"- GET http://127.0.0.1:{port}/state  -> View current state\n"
        f"- POST http://127.0.0.1:{port}/step  -> Take action. Payload strictly JSON: "
        f'{{"action": "<ACTION_NAME>", "predicted_outcome": "<prediction text>"}}\n\n'
        f"Valid Actions: {valid_actions_str}.\n\n"
        f"The server automatically logs every frame encountered to: {session.log_path}\n"
        f"The structured cross-task memory file (JSON) is located at: {global_memory_path}\n\n"
        f"--- WORKFLOW AND SKILLS ---\n{skills_text}\n---------------------------\n\n"
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

def get_plugin_environment(env_name: str) -> BaseEnvironment:
    """Dynamic Plugin Loader"""
    if env_name == "state_machine":
        from plugins.state_machine import StateMachineEnvironment
        return StateMachineEnvironment()
    raise ValueError(f"Unknown environment plugin '{env_name}'.")

def main():
    parser = argparse.ArgumentParser(description="Recursive Reasoning Framework Harness")
    parser.add_argument("--env", default="state_machine", help="Environment plugin to load (e.g., state_machine)")
    parser.add_argument("--tasks", nargs="*", default=["task_1_blue", "task_2_green"], help="List of task IDs")
    parser.add_argument("--results", default="results.json", help="Destination summary file")
    parser.add_argument("--max-actions", type=int, default=100, help="Max actions per task")
    parser.add_argument("--timeout", type=int, default=300, help="Agent timeout per task in seconds")
    parser.add_argument("--port", type=int, default=None, help="Force specific server port")
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    MEMORY_DIR.mkdir(exist_ok=True)

    if not SKILLS_TEMPLATE.exists():
        print(f"ERROR: {SKILLS_TEMPLATE} not found.", file=sys.stderr)
        sys.exit(1)

    env = get_plugin_environment(args.env)
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
                "duration_sec": round(duration, 2), "outcome_log": session.outcome_log,
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

