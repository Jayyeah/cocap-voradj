from pathlib import Path

import numpy as np
import pytest
import torch

from cocap_voradj.training.numeric_validation import assert_finite_numeric_tree


def test_accepts_numeric_tensors_arrays_scalars_and_nested_metadata():
    value = {
        "tensor": torch.tensor([0.0, 1.0]),
        "arrays": [
            np.asarray([1.0, -2.5], dtype=np.float32),
            np.asarray([1, 2], dtype=np.int64),
        ],
        "nested": ({"scalar": np.float64(3.0)}, 7, True),
        "metadata": {"phase": "capture", "note": None, "raw": b"ok"},
    }
    assert_finite_numeric_tree(value)


def test_object_ndarray_recurses_numeric_children_but_ignores_metadata():
    value = np.empty(5, dtype=object)
    value[:] = [
        "coverage",
        None,
        {"reward": np.asarray([1.0, 2.0], dtype=np.float32)},
        [torch.tensor(3.0), "tag"],
        (np.float64(4.0),),
    ]
    assert_finite_numeric_tree(value)

    value[2] = {"reward": np.asarray([1.0, np.nan], dtype=np.float32)}
    with pytest.raises(FloatingPointError, match=r"root\.flat\[2\].*reward"):
        assert_finite_numeric_tree(value)


@pytest.mark.parametrize(
    "value",
    [
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
        np.asarray([0.0, np.nan]),
        np.asarray([0.0, np.inf]),
        float("nan"),
        float("-inf"),
        complex(0.0, float("inf")),
    ],
)
def test_rejects_true_nan_and_inf(value):
    with pytest.raises(FloatingPointError):
        assert_finite_numeric_tree(value)


def test_handles_recursive_metadata_containers():
    recursive = []
    recursive.append(recursive)
    value = {"recursive": recursive, "metadata": np.asarray(["a", "b"])}
    assert_finite_numeric_tree(value)


def test_real_normsense_transition_checks_numeric_data_and_skips_phase_metadata(
    tmp_path: Path,
):
    from tools import launch_normsense_v2_formal as launch
    from tools import preflight_forward_final_scratch_20260914 as scratch
    from tools import train_forward_final_ppo_20260909 as production

    seed = 2026091501
    scratch.seed_all(seed)
    contract = scratch.load_contract()
    trainer = launch.make_trainer(contract, seed, "cpu")
    stream = launch.make_stream("pure_capture", seed, tmp_path, 1.0)
    row, episode = production.collect_transition(trainer, stream)

    assert episode is None
    assert row["gradient_phase"].dtype.kind == "U"
    assert row["local_obs"]["self"].dtype == np.float32
    assert_finite_numeric_tree(row, path="transition[0]")
