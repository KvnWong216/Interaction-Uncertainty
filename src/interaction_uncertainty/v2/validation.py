"""Small dependency-free validators for versioned JSON contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

import numpy as np


def require_exact_keys(
    payload: Mapping[object, object],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    location: str,
) -> None:
    """Reject missing, non-string, and unknown JSON object keys."""

    if any(not isinstance(key, str) for key in payload):
        raise TypeError(f"{location} keys must be strings")
    observed = set(payload)
    missing = required - observed
    unknown = observed - required - optional
    if missing:
        raise ValueError(f"{location} is missing required fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{location} contains unknown fields: {sorted(unknown)}")


def finite_number(
    value: object,
    *,
    location: str,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    """Parse one numeric scalar without accepting booleans or strings."""

    if isinstance(value, bool | np.bool_) or isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{location} must be a number, not {type(value).__name__}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{location} must be a numeric scalar") from exc
    if not isfinite(result):
        raise ValueError(f"{location} must be finite")
    if minimum is not None:
        invalid = result <= minimum if exclusive_minimum else result < minimum
        if invalid:
            operator = ">" if exclusive_minimum else ">="
            raise ValueError(f"{location} must be {operator} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{location} must be <= {maximum}")
    return result


def strict_integer(
    value: object,
    *,
    location: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Parse one integral scalar without truncation or boolean coercion."""

    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise TypeError(f"{location} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{location} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{location} must be <= {maximum}")
    return result


def strict_string(value: object, *, location: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{location} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{location} must be non-empty")
    return value


def strict_array(value: object, *, location: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{location} must be an array")
    return value
