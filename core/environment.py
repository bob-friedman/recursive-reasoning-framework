from typing import Any, Dict, List, Optional


class BaseEnvironment:
    """
    Domain-agnostic Base Environment Contract.
    Any custom environment plugin must implement these methods.

    Extensibility notes (v1.1):
    - Subclasses may override `domain`, `get_description()` and
      `get_observation_schema()` for richer metadata without breaking
      the minimal contract (get_valid_actions/start/step).
    - The harness discovers plugins dynamically via importlib; no
      hard-coded registration required.
    """

    # Optional class-level metadata — subclasses should override
    domain: str = "generic"
    description: str = "Base environment — override in subclass"

    def get_valid_actions(self) -> List[str]:
        raise NotImplementedError

    def start(self, task_id: str) -> Dict[str, Any]:
        """Initialize the environment and return the first observation."""
        raise NotImplementedError

    def step(self, action_payload: Dict[str, Any]) -> tuple[Dict[str, Any], bool, bool]:
        """
        Accept an arbitrary JSON dictionary action.
        Returns: (observation_dict, is_done, is_win)
        """
        raise NotImplementedError

    def get_metrics(self) -> Dict[str, Any]:
        return {}

    # --- Optional extensibility hooks (default implementations) ---

    def get_description(self) -> str:
        """Human-readable description for registry / --list-envs."""
        return getattr(self, "description", self.__class__.__doc__ or "")

    def get_domain(self) -> str:
        """Domain tag used for memory filtering (e.g., 'state_machine')."""
        return getattr(self, "domain", "generic")

    def validate_action(self, action_payload: Dict[str, Any]) -> Optional[str]:
        """
        Optional validation. Return error string if invalid, None if valid.
        Default checks against get_valid_actions().
        """
        try:
            valid = self.get_valid_actions()
        except NotImplementedError:
            return None  # cannot validate without implementation
        action = str(action_payload.get("action", "")).upper()
        if action not in valid:
            return f"Invalid action '{action}'. Must be one of {valid}"
        return None

    def get_observation_schema(self) -> Dict[str, Any]:
        """Optional JSON-schema for observations — used for docs/validation."""
        return {"type": "object"}
