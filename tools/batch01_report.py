#!/usr/bin/env python3
"""Publish a bounded, read-only scientific snapshot of current provisional runs.

Writes central reports only. Never starts jobs, changes grants, consumes final
seeds, promotes checkpoints, or edits source-locked training worktrees.
"""
import csv
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from batch01_status import DIRECTORY, ROOT, inspect_runs, read
from batch01_v3r2_commander import fp, hf, write

LABELS = ("p1-control", "p1-treatment", "r1", "n1", "t1", "c0")
OUT = DIRECTORY / "commander_v3r2/status_20261010"
REPORT = ROOT / "docs/experiments/BATCH01_STATUS_20261010_ZH.md"
TZ = ZoneInfo("Asia/Shanghai")


def git_receipt(item):
    label, root = item
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/" + branch], cwd=root, text=True, timeout=45).split()[0]
    if head != remote:
        raise ValueError(f"{label} local/remote HEAD differs; preserve both histories")
    if subprocess.check_output(["git", "status", "--porcelain", "--", "src", "configs"], cwd=root, text=True).strip():
        raise ValueError(f"{label} scientific source/config is dirty")
    return {"line": label, "branch": branch, "local_head": head, "remote_head": remote, "matched": True}


def last_metrics(output):
    with (output / "metrics.jsonl").open("rb") as f:
        f.seek(max(0, f.seek(0, 2) - 65536))
        lines = f.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            pass
    raise ValueError("no complete PPO metrics record")


def snapshot():
    OUT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(TZ)
    state = read(DIRECTORY / "batch_state.json")
    _, registered = inspect_runs()
    all_runs = {r["run_id"]: r for r in registered}
    current = {r["variant"]: r for r in registered if r.get("execution_mode") == "PROVISIONAL_LONG" and not r.get("scientific_quarantine") and r["source_candidate_sha"] == state["core"]["candidate_sha"]}
    if set(current) != set(LABELS):
        raise ValueError("current six-run set incomplete")
    receipts = [("core", Path(state["core"]["observed_worktree"])), ("qa", ROOT.parent / "terl-mappo-batch01-qa-20261009")]
    receipts += [(label, Path(current[label]["worktree"])) for label in LABELS]
    with ThreadPoolExecutor(max_workers=4) as pool:
        remotes = list(pool.map(git_receipt, receipts))
    summaries, curves, seeds_seen = [], [], {}
    for label in LABELS:
        row = current[label]
        output = Path(row["output_root"])
        manifest = read(output / "manifest.json")
        delta = read(row["delta_path"])
        if fp(delta) != row["delta_hash"] or manifest["delta_manifest"] != delta:
            raise ValueError(f"{label} delta binding changed")
        for change in delta["source_changes"]:
            if hf(Path(row["worktree"]) / change["path"]) != change["after"]:
                raise ValueError(f"{label} declared source delta changed")
        cp = row["checkpoint"]
        side = read(Path(cp["path"]).with_suffix(".batch01.json"))
        if hf(cp["path"]) != cp["sha256"] or side["checkpoint_sha256"] != cp["sha256"] or side["manifest_fingerprint"] != fp(manifest) or side["steps"] != cp["step"]:
            raise ValueError(f"{label} checkpoint/sidecar/manifest mismatch")
        reports = {}
        previous = all_runs.get(row.get("previous_run_id"))
        if previous:
            if previous.get("scientific_quarantine") or previous["source_candidate_sha"] != row["source_candidate_sha"]:
                raise ValueError("cannot reuse quarantined or other-candidate pilot")
        for source in ([previous] if previous else []) + [row]:
            for item in source.get("evaluation", []):
                path = Path(item["output"])
                if path.exists():
                    report = read(path)
                    if report["seed_domain"] != "screen" or report["status"] != "PROVISIONAL_SCREEN" or report["evaluation_protocol"] != manifest["evaluation_protocol"]:
                        raise ValueError("only same-protocol provisional screens may enter this report")
                    reports[report["steps"]] = (path, report)
        point_rows = []
        for step, (path, report) in sorted(reports.items()):
            target = OUT / "screens" / label / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            data = path.read_bytes()
            if target.exists() and target.read_bytes() != data:
                raise ValueError("refuse overwrite of a different archived evaluation")
            if not target.exists():
                target.write_bytes(data)
            expected_seed = report["evaluation_protocol"]["screen"]["seed_base"] + report["evaluation_protocol"]["scene_offsets"][report["scene"]]
            physical_seeds = list(range(expected_seed, expected_seed + report["episodes_per_mode"]))
            for mode in report["modes"]:
                episodes = [e for e in report["episodes"] if e["mode"] == mode]
                if [e["seed"] for e in episodes] != physical_seeds:
                    raise ValueError("actual screen seeds do not match bound scene protocol")
                for episode in episodes:
                    if mode == "sample" and episode["sample_action_seed"] != episode["seed"] + report["evaluation_protocol"]["sample_torch_offset"]:
                        raise ValueError("sample action RNG offset mismatch")
                    key = (report["scene"], episode["seed"])
                    if seeds_seen.setdefault(key, episode["initial_fingerprint"]) != episode["initial_fingerprint"]:
                        raise ValueError("paired physical initial-state fingerprint differs")
                aggregate = report["modes"][mode]
                point = {"line": label, "step": step, "mode": mode, "episodes": aggregate["episodes"], "scene": report["scene"], "seed_base": report["seed_base"], **{k: aggregate[k] for k in ("normal_capture_rate", "collision_rate", "ring2_rate", "ring3_rate", "strict_geometry_rate")}, "checkpoint_sha256": report["checkpoint_sha256"], "source_evaluation": str(path), "evaluation_sha256": hf(path), "cloud_evaluation": str(target.relative_to(ROOT))}
                point_rows.append(point)
                curves.append(point)
        latest = max(reports)
        modes = reports[latest][1]["modes"]
        best = {mode: max((p for p in point_rows if p["mode"] == mode), key=lambda p: (p["normal_capture_rate"], -p["collision_rate"], -p["step"])) for mode in modes}
        metrics = last_metrics(output)
        if label.startswith("p1-") and metrics["live_target_kl"] != (.02 if label == "p1-control" else .01):
            raise ValueError("live P1 target KL differs from authorized branch")
        remaining = row["authorized_end_step"] - row["step"]
        rate = row.get("decisions_per_second", 0)
        eta = timestamp + timedelta(seconds=remaining / rate) if row["alive"] and rate > 0 else None
        summaries.append({**{k: row.get(k) for k in ("variant", "run_id", "status", "alive", "pid", "physical_gpu", "gpu_uuid", "step", "update", "heartbeat", "authorized_end_step", "head", "branch", "delta_hash", "worktree", "output_root", "tmux", "disk_bytes", "decisions_per_second", "last_exception", "resource_resume_history")}, "checkpoint": cp, "live_target_kl": metrics["live_target_kl"], "source_and_checkpoint_hashes_verified": True, "training_eta_Asia_Shanghai": eta.isoformat() if eta else None, "screen_checkpoint_count": len(reports), "screen_episodes": sum(len(r["episodes"]) for _, r in reports.values()), "latest_screen_step": latest, "latest_screen": modes, "best_screen_observed": best, "formal_evidence": False})
    resource = {"GPU": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader"], text=True), "RAM": {x.split(":")[0]: int(x.split()[1]) * 1024 for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith(("MemAvailable:", "MemTotal:"))}}
    stat = os.statvfs("/home/yjq")
    resource["disk_free_bytes"] = stat.f_bavail * stat.f_frsize
    runtime = ROOT.parent / "batch01-commander-runtime"
    resource["supervisor"] = read(runtime / "supervisor.json")
    doc = {"schema": "batch01.provisional_status_snapshot.v1", "time_Asia_Shanghai": timestamp.isoformat(), "source_candidate_sha": state["core"]["candidate_sha"], "canonical_lock_sha256": state["core"]["canonical_lock_sha256"], "core_status": state["core"]["status"], "QA_status": state["qa"]["status"], "base_freeze_status": state["base_freeze_status"], "BASE_FROZEN": state["base_freeze_status"] == "BASE_FROZEN", "formal_RUNNING": 0, "retroactive_promotion": False, "final_consumed_by_report": False, "paired_screen_initial_states_verified": True, "best_screen_rule": "descriptive maximum normal_capture_rate; then minimum collision; then earliest step; no selection receipt or promotion", "ready_pending_authorized_runs": [], "blocked_next_runs": {"T1_Stage3": ["qualified Stage2 selection and Stage1 retention absent", "no Stage3 provisional budget registered"]}, "remotes": remotes, "resources": resource, "runs": summaries}
    if not doc["BASE_FROZEN"]:
        doc["blocked_next_runs"]["T1_Stage3"].insert(0, "independent same-candidate QA pending; BASE unfrozen")
    write(OUT / "snapshot.json", doc)
    with (OUT / "screen_curves.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curves[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(curves)
    plot(curves)
    render(doc)
    print(json.dumps({"time": doc["time_Asia_Shanghai"], "runs": len(summaries), "screen_points": len(curves), "completed": sum(not r["alive"] and r["step"] == r["authorized_end_step"] for r in summaries), "running": sum(r["alive"] for r in summaries), "report": str(REPORT)}))


def plot(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, sharey=True)
    for column, mode in enumerate(("argmax", "sample")):
        for label, color in (("p1-control", "#315c96"), ("p1-treatment", "#cb5b35")):
            points = [p for p in curves if p["line"] == label and p["mode"] == mode]
            for index, metric in enumerate(("normal_capture_rate", "collision_rate")):
                axes[index, column].plot([p["step"] / 1000 for p in points], [p[metric] * 100 for p in points], marker="o", label=label, color=color)
                axes[index, column].grid(alpha=.25)
                axes[index, column].set_ylim(0, 105)
        axes[0, column].set_title(mode)
        axes[1, column].set_xlabel("Joint decisions (thousands)")
    axes[0, 0].set_ylabel("Normal capture (%)")
    axes[1, 0].set_ylabel("Collision (%)")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("P1 paired fork, 775k to 1M — PROVISIONAL screens (n=10/20 per mode)")
    fig.tight_layout()
    fig.savefig(OUT / "p1_screen_curves.png", dpi=160)
    plt.close(fig)


def render(doc):
    link = "../../" + str(OUT.relative_to(ROOT))
    rows = doc["runs"]
    complete = sum(r["status"] == "PROVISIONAL_LONG_COMPLETE" for r in rows)
    running = sum(r["alive"] for r in rows)
    lines = ["# Batch01 运行状态与结果快照", "", f"快照：{doc['time_Asia_Shanghai']}（Asia/Shanghai）。{complete}条线完成授权预算与screen，{running}条训练进程仍在运行；全部属于`PROVISIONAL_LONG`。没有合格且尚未启动的已登记预算。", "", f"科学candidate：`{doc['source_candidate_sha']}`；canonical lock：`{doc['canonical_lock_sha256']}`。`{doc['core_status']} / {doc['QA_status']} / {doc['base_freeze_status']}`。同一Commander执行A3独立编写的测试，不等于独立审核签字；正式RUNNING=0，不能追认为正式证据。", "", "## 当前运行与checkpoint", "", "| 线 | 状态 | 当前PID / GPU | step / 授权终点 | update | 最新checkpoint step |", "|---|---|---|---|---:|---:|"]
    for r in rows:
        pid = f"{r['pid']} / {r['physical_gpu']}" if r["alive"] else "已退出"
        status = "PROVISIONAL运行" if r["alive"] else "PROVISIONAL预算与screen完成" if r["status"] == "PROVISIONAL_LONG_COMPLETE" else r["status"]
        lines.append(f"| {r['variant']} | {status} | {pid} | {r['step']:,} / {r['authorized_end_step']:,} | {r['update']:,} | {r['checkpoint']['step']:,} |")
    active = next(r for r in rows if r["variant"] == "r1")
    eta_note = f"R1粗估训练终点：{active['training_eta_Asia_Shanghai']}；训练结束后仍须等待末点screen完成。" if active["alive"] else "R1训练进程已退出，预算与screen状态见上表。"
    lines += ["", f"{eta_note}实际吞吐、完整checkpoint路径/hash、run_id、GPU UUID、历史PID、恢复回滚和异常详见[机器快照]({link}/snapshot.json)与[run registry](../../artifacts/2026-10-09_terl_mappo_batch01/run_registry.json)。本次六个最新checkpoint及sidecar/manifest绑定、各ARM声明源码delta重新hash通过。", "", "## 最新screen结果", "", "以下是screen而非final，不能用于宣称总体优越性。表内为deterministic/sample；C0 deterministic为mean，其余为argmax。两个模式共享物理初态，不能把20+20当成40个独立seed。", "", "| 线 | screen step | 每模式n | normal capture | collision |", "|---|---:|---|---|---|"]
    for r in rows:
        modes = list(r["latest_screen"].values())
        pair = lambda key: "/".join(f"{m[key]*100:.0f}%" for m in modes)
        lines.append(f"| {r['variant']} | {r['latest_screen_step']:,} | {'/'.join(str(m['episodes']) for m in modes)} | {pair('normal_capture_rate')} | {pair('collision_rate')} |")
    lines += ["", "## Best-observed与保持曲线", "", "best-observed按normal最高、collision最低、最早step记录，仅描述已观察screen，不生成selection receipt或promotion。n随预登记milestone为10或20，且多次观察存在选择偏差。", "", "| 线 | 模式 | best screen step | n | normal capture | collision |", "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        for mode, p in r["best_screen_observed"].items():
            lines.append(f"| {r['variant']} | {mode} | {p['step']:,} | {p['episodes']} | {p['normal_capture_rate']*100:.0f}% | {p['collision_rate']*100:.0f}% |")
    lines += ["", "P1两个分支已从同一775k完整状态配对到1M，live target KL分别0.02/0.01；末点treatment的normal capture高于control，属于本次单状态分叉的描述性结果。已覆盖775k之后全部25k screen点，同时保留best-observed与last，未创建selected/best promotion。", "", f"![P1配对screen保持曲线]({link}/p1_screen_curves.png)", "", f"[全部六线screen曲线CSV]({link}/screen_curves.csv)；[完整screen原件]({link}/screens/)按字节归档，机器快照列出每线总评估量。逐episode物理初态及sample seed offset已验证配对一致。", "", "## 启动与升级门禁", "", "T1 Stage2 100k已经完成首个review窗口，不把100k解释为必然失败。末点screen sample normal55%、collision40%，尚低于升级目标normal≥80%、collision≤20%；正式升级还须selection域每模式20局的Stage2资格及Stage1 retention，并须独立QA/冻结BASE。现在缺少上述合格报告，且尚无Stage3 PROVISIONAL预算，因此不启动Stage3，也不自动扩Stage2到200k。其它五线按各自已授权终点处理，未新增同名run。", "", "最终2076100900域保持盲态；screen2056100900、selection2066100900、sample offset100000、scene offsets0/1000/2000保持绑定，历史T0 final不变。", "", "资源检查只在启动/排队时决定是否可开启新任务；监督器运行期RAM/GPU/磁盘/租约阈值均为告警，不杀停已启动训练。源码、checkpoint、预算、QA门禁继续保留。训练器自身锁定源码中的2GiB allocation/8GiB输出/30GiB磁盘检查未热改，当前未触发。两卡仍有受保护的外部任务；没有停止或修改外部PID。", "", "## 云端与恢复入口", "", "Core、QA及六ARM的本地HEAD与远程HEAD逐一一致；完整分支与SHA见机器快照。中央文档、原始screen证据、曲线、资源恢复回执与DAG/run registry随本轮commit推送GitHub。checkpoint权重保留在服务器，仅同步路径/hash，不将大体积权重伪称已上传云端。", "", "```bash", "cd /home/yjq/rl/CoCap1/ac-master-dag-20260921", "python tools/batch01_status.py --current", "python tools/batch01_status.py --write --json", "python tools/batch01_report.py", "```", "", "持久监督器tmux：`batch01_commander_v3r2_long_supervisor`；恢复队列：`batch01_commander_resource_recovery`。CLI断连后可从中央registry、每run progress/manifest及runtime heartbeat恢复。NEXT WAKE-UP：R1后续25k checkpoint/screen与1M完成，或独立同SHA/lock QA签字到达；完成预算的线不自动续训。", ""]
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    snapshot()
