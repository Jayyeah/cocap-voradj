"""Type-safe finite-value validation for nested training data.

Training transitions contain both numeric leaves and observational metadata.
In particular, NumPy object arrays can hold strings, None, nested containers,
or numeric arrays. Treating every ndarray as numeric makes np.isfinite itself
raise before it can validate the numeric leaves.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def assert_finite_numeric_tree(value: Any, *, path: str = "root") -> None:
    """Raise FloatingPointError for non-finite numeric leaves in value.

    Numeric torch tensors, numeric NumPy arrays, and numeric scalars are
    checked. Mappings, lists, tuples, and object ndarrays are traversed.
    Non-numeric metadata (for example strings, bytes, and None) is ignored.
    """
    _assert_finite_numeric_tree(value, path=path, seen=set())


def _assert_finite_numeric_tree(value: Any, *, path: str, seen: set[int]) -> None:
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"non-finite tensor at {path}")
        return

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            identity = id(value)
            if identity in seen:
                return
            seen.add(identity)
            try:
                for index, child in enumerate(value.flat):
                    _assert_finite_numeric_tree(
                        child, path=f"{path}.flat[{index}]", seen=seen
                    )
            finally:
                seen.remove(identity)
        elif np.issubdtype(value.dtype, np.number):
            if not bool(np.isfinite(value).all()):
                raise FloatingPointError(
                    f"non-finite ndarray at {path} (dtype={value.dtype})"
                )
        return

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        try:
            for key, child in value.items():
                _assert_finite_numeric_tree(
                    child, path=f"{path}[{key!r}]", seen=seen
                )
        finally:
            seen.remove(identity)
        return

    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        try:
            for index, child in enumerate(value):
                _assert_finite_numeric_tree(
                    child, path=f"{path}[{index}]", seen=seen
                )
        finally:
            seen.remove(identity)
        return

    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"non-finite scalar at {path}")
        return

    if isinstance(value, (complex, np.complexfloating)):
        if not (math.isfinite(float(value.real)) and math.isfinite(float(value.imag))):
            raise FloatingPointError(f"non-finite scalar at {path}")
