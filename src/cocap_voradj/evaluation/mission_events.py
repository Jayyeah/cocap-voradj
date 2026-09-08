"""Target-indexed evaluation events, independent of reward and policy inputs.

Times are decision-transition boundaries (step 0 is reset). Uncompleted
opportunities remain explicit outcomes; cancelled events are not successes.
The support trigger uses targets directly visible to an observed teammate:
this is evaluator-only information, not an enemy-coordinate message.
"""
from __future__ import annotations
import math
import numpy as np

SCHEMA = "same-target-mission-events-v1"


def snapshot(env, observations):
    data = {"targets": {}, "active_agents": set(), "direct": {}, "friends": {}}
    radius = float(env.env_cfg.get("capture_distance", env.reward_cfg.get("r_e", 8.0)))
    for i, obs in enumerate(observations):
        if obs is None or env.pursuers[i].deactivated:
            continue
        data["active_agents"].add(i)
        for field, type_id, entities in (("direct", 2, env.evaders), ("friends", 1, env.pursuers)):
            rows = obs["evaders" if field == "direct" else "pursuers"]
            mask = np.asarray(obs["masks"])[np.asarray(obs["types"]) == type_id].astype(bool)
            visible = set()
            for token in rows[mask]:
                for j, entity in enumerate(entities):
                    if entity.deactivated or (field == "friends" and i == j):
                        continue
                    world = str(env.per_cfg.get("observation_frame", "robot")) in {"world", "world_frame"}
                    p = np.array([entity.x, entity.y])
                    rel = p - np.array([env.pursuers[i].x, env.pursuers[i].y]) if world else env._robot_frame(env.pursuers[i], p, False)
                    if np.allclose(rel / env._distance_scale(), token[:2], rtol=1e-5, atol=1e-6):
                        visible.add(j)
            data[field][i] = visible
    for j, evader in enumerate(env.evaders):
        if evader.deactivated:
            continue
        participants, angles = set(), []
        for i in data["active_agents"]:
            delta = np.array([env.pursuers[i].x-evader.x, env.pursuers[i].y-evader.y])
            if np.linalg.norm(delta) <= radius:
                participants.add(i); angles.append(math.atan2(delta[1], delta[0]) % (2*math.pi))
        angles.sort()
        gaps = [(angles[(k+1) % len(angles)]-a) % (2*math.pi) for k,a in enumerate(angles)]
        geometric = len(participants) >= int(env.reward_cfg.get("k_required", 3)) and bool(gaps) and max(gaps) <= float(env.reward_cfg.get("max_angle_gap", math.pi)) and max(gaps) <= min(gaps)*float(env.reward_cfg.get("max_angle_ratio", 3.0))
        data["targets"][j] = {"participants": participants, "geometric": bool(geometric)}
    return data


class MissionEventTracker:
    def __init__(self, decision_dt):
        self.dt = float(decision_dt)
        self.open = {}
        self.events = []
        self.previous_triggers = set()

    def observe(self, state, step, capture_events=()):
        # Env removes captured targets before packing observations. Reconstruct
        # the terminal geometry from the authoritative capture event snapshot.
        targets = {j: dict(v) for j,v in state["targets"].items()}
        removed = set()
        for event in capture_events:
            j = int(event["evader_id"]); removed.add(j)
            targets[j] = {"participants": set(event["participants"]), "geometric": event.get("capture_type") != "stationary"}
        triggers, complete = set(), set()
        for j, target in targets.items():
            participants = target["participants"]
            if len(participants) >= 2:
                for metric in ("capture_region_2_to_3", "capture_region_2_to_geometry"):
                    key = (metric, j, None); triggers.add(key)
                    if (len(participants) >= 3 if metric.endswith("to_3") else target["geometric"]):
                        complete.add(key)
        for i in state["active_agents"]:
            neighbor_targets = set().union(*(state["direct"].get(f, set()) for f in state["friends"].get(i, set())))
            for j in neighbor_targets - state["direct"].get(i, set()):
                triggers.add(("support_to_direct", j, i))
                triggers.add(("support_to_capture_region", j, i))
        # A direct detector entering the real capture region completes the
        # earlier request even though it no longer satisfies the trigger.
        for key in set(self.open) | triggers:
            metric, j, i = key
            if metric == "support_to_direct" and j in state["direct"].get(i, set()):
                complete.add(key)
            if metric == "support_to_capture_region" and i in targets.get(j, {}).get("participants", set()):
                complete.add(key)
        for key in triggers - self.previous_triggers:
            if key not in self.open:
                self.open[key] = int(step)
        for key in list(self.open):
            metric, j, i = key
            if key in complete:
                self._end(key, step, "completed")
            elif j in removed or j not in targets:
                self._end(key, step, "target_removed")
            elif i is not None and i not in state["active_agents"]:
                self._end(key, step, "agent_inactive")
            elif metric.startswith("capture_region") and key not in triggers:
                self._end(key, step, "coalition_lost")
            elif metric.startswith("support") and key not in triggers and j not in state["direct"].get(i, set()):
                self._end(key, step, "support_lost")
        self.previous_triggers = triggers

    def _end(self, key, step, outcome):
        start = self.open.pop(key)
        self.events.append({"metric": key[0], "target_id": key[1], "agent_id": key[2], "start_step": start, "end_step": int(step), "duration_steps": int(step)-start, "duration_seconds": (int(step)-start)*self.dt, "outcome": outcome, "left_censored": start == 0, "right_censored": outcome == "episode_end"})

    def finish(self, step):
        for key in list(self.open):
            self._end(key, step, "episode_end")
        return {"schema": SCHEMA, "decision_dt": self.dt, "events": self.events, "summary": summarize_events(self.events)}


def summarize_events(events):
    result = {}
    for metric in sorted({e["metric"] for e in events}):
        rows = [e for e in events if e["metric"] == metric]
        complete = [e for e in rows if e["outcome"] == "completed"]
        known_start = [e for e in complete if not e["left_censored"]]
        result[metric] = {"completed_known_start_mean_seconds": float(np.mean([e["duration_seconds"] for e in known_start])) if known_start else None, "completed_known_start_count": len(known_start), "opportunities": len(rows), "completed": len(complete), "completion_rate": len(complete)/len(rows), "outcomes": {s: sum(e["outcome"] == s for e in rows) for s in sorted({e["outcome"] for e in rows})}, "completed_mean_seconds": float(np.mean([e["duration_seconds"] for e in complete])) if complete else None, "left_censored": sum(e["left_censored"] for e in rows)}
    return result
