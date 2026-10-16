"""Pure-Python job intake contracts (worker and persistence added later)."""

from .inputs import (
    InputKind, InputLimits, InputSource, InputValidationError, ValidatedInput,
    validate_input,
)

__all__ = (
    "InputKind", "InputLimits", "InputSource", "InputValidationError",
    "ValidatedInput", "validate_input",
)
