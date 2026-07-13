"""Structured failure reasons shared by the command-line pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any


class FailureCode(str, Enum):
    CONFIG_INVALID = "config_invalid"
    ASSET_MISSING = "asset_missing"
    ASSET_INVALID = "asset_invalid"
    FRAME_MISSING = "frame_missing"
    NAME_NOT_UNIQUE = "name_not_unique"
    IK_UNREACHABLE = "ik_unreachable"
    JOINT_LIMIT = "joint_limit_violation"
    COLLISION = "collision"
    NUMERICAL_INSTABILITY = "numerical_instability"
    PHYSICS_UNAVAILABLE = "physics_unavailable"
    OUTPUT_FAILURE = "output_failure"


class PipelineError(RuntimeError):
    """An expected pipeline failure with a machine-readable reason."""

    def __init__(
        self,
        code: FailureCode | str,
        message: str,
        *,
        stage: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = FailureCode(code)
        self.stage = stage
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "stage": self.stage,
            "message": str(self),
            "details": self.details,
        }

