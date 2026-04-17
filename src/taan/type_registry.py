"""
Task type registry for TAAN-GRPO.

Centralises the mapping from task type string identifiers to metadata,
reward function factories, and sampling weights so that other modules
have a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class TaskTypeInfo:
    """Metadata for a single task type."""

    type_id: str
    description: str = ""
    reward_fn_name: str = ""
    reward_config: Dict[str, Any] = field(default_factory=dict)
    signal_type: str = "rule"   # "rule" | "model"
    weight: float = 1.0        # relative sampling weight


class TypeRegistry:
    """Registry that maps type_id strings to :class:`TaskTypeInfo` objects.

    Example usage::

        registry = TypeRegistry()
        registry.register(TaskTypeInfo(type_id="math", reward_fn_name="math_reward"))
        info = registry["math"]
    """

    def __init__(self) -> None:
        self._registry: Dict[str, TaskTypeInfo] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, info: TaskTypeInfo) -> None:
        """Register a task type.

        Args:
            info: :class:`TaskTypeInfo` instance to register.

        Raises:
            ValueError: If the type_id is already registered.
        """
        if info.type_id in self._registry:
            raise ValueError(
                f"Task type '{info.type_id}' is already registered. "
                "Use update() to overwrite."
            )
        self._registry[info.type_id] = info

    def update(self, info: TaskTypeInfo) -> None:
        """Register or overwrite a task type."""
        self._registry[info.type_id] = info

    def register_from_config(self, task_types_config: Dict[str, Any]) -> None:
        """Bulk-register types from a parsed YAML/dict config.

        Expected format (matching ``configs/task_types.yaml``)::

            {
                "math": {
                    "description": "...",
                    "reward": "math_reward",
                    "reward_config": {},
                    "signal_type": "rule",
                },
                ...
            }
        """
        for type_id, cfg in task_types_config.items():
            info = TaskTypeInfo(
                type_id=type_id,
                description=cfg.get("description", ""),
                reward_fn_name=cfg.get("reward", ""),
                reward_config=cfg.get("reward_config", {}),
                signal_type=cfg.get("signal_type", "rule"),
                weight=cfg.get("weight", 1.0),
            )
            self.update(info)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __getitem__(self, type_id: str) -> TaskTypeInfo:
        if type_id not in self._registry:
            raise KeyError(f"Unknown task type: '{type_id}'")
        return self._registry[type_id]

    def get(self, type_id: str, default: Optional[TaskTypeInfo] = None) -> Optional[TaskTypeInfo]:
        return self._registry.get(type_id, default)

    def __contains__(self, type_id: str) -> bool:
        return type_id in self._registry

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def type_ids(self) -> List[str]:
        """Sorted list of all registered type IDs."""
        return sorted(self._registry.keys())

    def weights(self) -> Dict[str, float]:
        """Return {type_id: weight} mapping."""
        return {tid: info.weight for tid, info in self._registry.items()}

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        ids = ", ".join(self.type_ids)
        return f"TypeRegistry(types=[{ids}])"


# ---------------------------------------------------------------------------
# Module-level default registry
# ---------------------------------------------------------------------------

_default_registry = TypeRegistry()

# Pre-register common types so that imports work without explicit configuration
for _type_id, _desc, _reward, _signal in [
    ("math",    "Mathematical reasoning",         "math_reward",  "rule"),
    ("code",    "Code generation & execution",    "code_reward",  "rule"),
    ("writing", "Open-ended writing",             "model_reward", "model"),
    ("chat",    "Conversational dialogue",        "model_reward", "model"),
    ("safety",  "Safety-critical prompts",        "math_reward",  "rule"),
]:
    _default_registry.update(
        TaskTypeInfo(
            type_id=_type_id,
            description=_desc,
            reward_fn_name=_reward,
            signal_type=_signal,
        )
    )


def get_default_registry() -> TypeRegistry:
    """Return the module-level default :class:`TypeRegistry`."""
    return _default_registry
