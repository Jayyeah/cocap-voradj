from __future__ import annotations

from pathlib import Path

from open_encirclement_ctde.training import load_config


ROOT = Path(__file__).resolve().parents[2]


def test_six_formal_configs_are_matched_and_frozen() -> None:
    paths = sorted((ROOT / "configs" / "open_encirclement_ctde").glob("*.json"))
    assert len(paths) == 6
    configs = [load_config(path) for path in paths]
    assert {(cfg["algorithm"], cfg["seed"]) for cfg in configs} == {
        (algorithm, seed) for algorithm in ("mappo", "maddpg") for seed in (1, 2, 3)
    }
    for config in configs:
        assert config["contract"] == "corrected_roundup_v1"
        assert config["num_env_steps"] == 150_000
        assert config["n_envs"] == 8
        assert config["checkpoint_interval"] == 25_000
        assert config["eval_interval"] == 25_000
        assert config["eval_episodes"] == 20
    for config in (cfg for cfg in configs if cfg["algorithm"] == "mappo"):
        algo = config["mappo"]
        assert (algo["hidden_size"], algo["layer_N"]) == (64, 1)
        assert (algo["lr"], algo["critic_lr"]) == (5e-4, 5e-4)
        assert algo["ppo_epoch"] == 15
        assert algo["clip_param"] == 0.2
        assert algo["entropy_coef"] == 0.01
        assert (algo["gamma"], algo["gae_lambda"]) == (0.99, 0.95)
        assert algo["max_grad_norm"] == 10.0
    for config in (cfg for cfg in configs if cfg["algorithm"] == "maddpg"):
        algo = config["maddpg"]
        assert algo["hidden_dim"] == 128
        assert (algo["actor_lr"], algo["critic_lr"]) == (1e-4, 3e-3)
        assert (algo["gamma"], algo["tau"]) == (0.99, 0.01)
        assert (algo["replay_capacity"], algo["batch_size"]) == (1_000_000, 256)
        assert (algo["actor_scheduler_step"], algo["actor_scheduler_gamma"]) == (1_000, 0.8)
        assert (algo["critic_scheduler_step"], algo["critic_scheduler_gamma"]) == (5_000, 0.33)


def test_supervisor_lanes_cover_every_config_once() -> None:
    import importlib.util
    import sys

    path = ROOT / "tools" / "open_encirclement_ctde" / "supervise_l0.py"
    spec = importlib.util.spec_from_file_location("open_ctde_supervisor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    scheduled = [name for names in module.LANES.values() for name in names]
    expected = [path.name for path in (ROOT / "configs" / "open_encirclement_ctde").glob("*.json")]
    assert sorted(scheduled) == sorted(expected)
    assert len(scheduled) == len(set(scheduled)) == 6


def test_open_baseline_namespace_does_not_import_cocap_legacy_features() -> None:
    forbidden = ("LegacyVorAdj", "K10", "support_coverage", "post_capture")
    for path in (ROOT / "src" / "open_encirclement_ctde").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden), path
