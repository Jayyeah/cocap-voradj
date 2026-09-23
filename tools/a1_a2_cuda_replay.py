#!/usr/bin/env python3
"""Replay exact captured actor weights/rows; compare CPU/CUDA without hooks."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from tools.a1_a2_cuda_audit import OUT, backend_info, mask_info, tensor_info, write_json
import torch
import torch.nn.functional as F
from cocap_voradj.models.shared_local_ac import SharedLocalActor, SharedLocalACNetworkConfig
from cocap_voradj.training.shared_local_ac import state_hash


class Recorder:
    """Wrap callables, not hooks, preserving PyTorch's fused-path eligibility."""

    def __init__(self):
        self.values = {}
        self.events = []
        self.patches = []

    def record(self, name, value):
        if isinstance(value, torch.Tensor):
            key = f"{len(self.events):04d}:{name}"
            self.values[key] = value.detach().cpu().clone()
            self.events.append({"key": key, "name": name, **tensor_info(value)})
        elif isinstance(value, dict):
            for k, v in value.items():
                self.record(name + "." + k, v)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                self.record(name + f".{i}", v)

    def patch(self, obj, name, wrapper):
        self.patches.append((obj, name, getattr(obj, name)))
        setattr(obj, name, wrapper)

    @contextmanager
    def installed(self, actor):
        for name, module in actor.named_modules():
            # Container outputs are redundant; Transformer layers expose fused
            # inputs/outputs even when their internal Python modules are bypassed.
            if not name or isinstance(module, (torch.nn.ModuleDict, torch.nn.Sequential, torch.nn.ModuleList)):
                continue
            original = module.forward
            def forward(*a, _old=original, _name=name, **kw):
                if "transformer.layers." in _name and _name.count(".") == 3:
                    self.record(_name + ".input", a[0])
                    self.record(_name + ".padding_mask", kw.get("src_key_padding_mask"))
                value = _old(*a, **kw)
                self.record(_name, value)
                return value
            self.patch(module, "forward", forward)
        for name in ("features", "decision_feature"):
            original = getattr(actor.encoder, name)
            def method(*a, _old=original, _name=name, **kw):
                value = _old(*a, **kw)
                self.record("encoder." + _name, value)
                return value
            self.patch(actor.encoder, name, method)
        original = torch._transformer_encoder_layer_fwd
        def fused(*a, **kw):
            self.record("fused_transformer.input", a[0])
            value = original(*a, **kw)
            self.record("fused_transformer.output", value)
            return value
        self.patch(torch, "_transformer_encoder_layer_fwd", fused)
        try:
            yield self
        finally:
            for obj, name, original in reversed(self.patches):
                setattr(obj, name, original)


@torch.no_grad()
def trace(actor, obs):
    rec = Recorder()
    rec.record("packed_observation", obs)
    with rec.installed(actor):
        logits = actor.logits(obs)
        rec.record("actor.logits", logits)
        rec.record("actor.softmax", torch.softmax(logits, dim=-1))
        rec.record("SAC.log_softmax", F.log_softmax(logits, dim=-1))
        rec.record("behavior_mixture_epsilon_0.6", 0.4 * torch.softmax(logits, dim=-1) + 0.6 / 9)
    return rec


@torch.no_grad()
def attention_probe(actor, obs, actual_padding_mask):
    """Isolate the native MHA inside layer 0 with the captured tensor/mask.

    This is separately labeled decomposition, not the production forward.
    Native MHA uses the same norm1 and in/out projections as the fused layer.
    """
    e = actor.encoder
    emb = [e.encoders["self"](obs["self"]).unsqueeze(1)]
    emb.extend(e.encoders[k](obs[k]) for k in ("pursuers", "evaders", "obstacles"))
    tokens = torch.cat(emb, 1) + e.type_embedding(obs["types"].long())
    layer = e.transformer.layers[0]
    x = layer.norm1(tokens)
    # Use the mask actually passed to the production layer, including the
    # internal placeholder repair when testing the fixed implementation.
    canonical_mask = actual_padding_mask.to(device=x.device, dtype=x.dtype)
    mask = canonical_mask.bool()
    mha = layer.self_attn
    y, weights = torch._native_multi_head_attention(x, x, x, mha.embed_dim, mha.num_heads, mha.in_proj_weight, mha.in_proj_bias, mha.out_proj.weight, mha.out_proj.bias, canonical_mask, True, False, 1)
    qkv = F.linear(x, mha.in_proj_weight, None)
    q, k, v = torch.ops.aten._transform_bias_rescale_qkv(qkv, mha.in_proj_bias, mha.num_heads)
    b, h, t, d = q.shape
    scores = torch.bmm(q.reshape(b*h,t,d), k.reshape(b*h,t,d).transpose(1,2)).view(b,h,t,t)
    probabilities = torch.ops.aten._masked_softmax(scores, canonical_mask.bool(), -1, 1)
    masked_scores = scores.masked_fill(mask[:, None, None, :], float("-inf"))
    residual = tokens + y
    norm2 = layer.norm2(residual)
    ff = layer.linear2(layer.activation(layer.linear1(norm2)))
    decomposed = residual + ff
    fused = layer(tokens, src_key_padding_mask=canonical_mask)
    both = torch.isfinite(fused) & torch.isfinite(decomposed)
    return {
        "evidence_kind": "separate_native_attention_decomposition_no_backend_switch",
        "stages": {name: tensor_info(value) for name, value in [("tokens",tokens),("norm1",x),("qkv_projection",qkv),("q",q),("k",k),("v",v),("qk_scores",scores),("masked_scores_expected_negative_inf",masked_scores),("masked_softmax",probabilities),("native_attention_weights",weights),("native_attention_output",y),("attention_residual",residual),("norm2",norm2),("ff",ff),("decomposed_layer",decomposed),("production_fused_layer",fused)]},
        "bad_rows_attention": (~torch.isfinite(y)).flatten(1).any(1).nonzero().flatten().cpu().tolist(),
        "bad_rows_softmax": (~torch.isfinite(probabilities)).flatten(1).any(1).nonzero().flatten().cpu().tolist(),
        "fused_vs_decomposed_finite_pattern_equal": bool(torch.equal(torch.isfinite(fused), torch.isfinite(decomposed))),
        "fused_vs_decomposed_max_abs_common_finite": float((fused[both]-decomposed[both]).abs().max()) if bool(both.any()) else None,
    }


@torch.no_grad()
def repeats(actor, obs, bad_row, counts):
    row = {k: v[bad_row:bad_row+1] for k, v in obs.items()}
    variants = {"single_row": (row, counts), "original_full_batch": (obs,[10]), "reversed_full_batch": ({k: v.flip(0) for k,v in obs.items()},[10]), "duplicate_bad_row": ({k: v.repeat(4,*([1]*(v.ndim-1))) for k,v in row.items()},[10])}
    result = {}
    for name, (batch, checkpoints) in variants.items():
        result[name] = []
        for count in checkpoints:
            failures = 0
            first = None
            first_values = None
            same_pattern = True
            start = time.monotonic()
            for i in range(count):
                logits = actor.logits(batch)
                pattern = torch.isfinite(logits).cpu()
                bad = not bool(pattern.all())
                if bad:
                    failures += 1
                    if first is None:
                        first = i + 1
                if first_values is None:
                    first_values = pattern
                else:
                    same_pattern &= torch.equal(first_values,pattern)
                # Keep shared GPU utilization bounded during repeated replay.
                time.sleep(0.008 if name == "single_row" else 0.04)
            result[name].append({"runs": count, "failure_count": failures, "first_failing_iteration_1based": first, "finite_pattern_deterministic": bool(same_pattern), "seconds": time.monotonic()-start})
    return result


def run(args):
    torch.set_num_threads(1)
    payload = torch.load(args.artifact, map_location="cpu", weights_only=False)
    from tools.a1_a2_cuda_audit import load_module
    original_module = load_module("audit_original_encoder", OUT / "source_snapshot/a2/src/cocap_voradj/models/shared_local_ac.py")
    model_module = original_module if args.implementation == "original" else sys.modules["cocap_voradj.models.shared_local_ac"]
    config = model_module.SharedLocalACNetworkConfig(**payload["network_config"])
    cpu = model_module.SharedLocalActor(config).eval()
    cpu.load_state_dict(payload["state_dict"])
    assert state_hash(cpu) == payload["manifest"]["model_hash"]
    cuda = model_module.SharedLocalActor(config).eval().to("cuda:0")
    cuda.load_state_dict(payload["state_dict"])
    obs_cpu = payload["obs"]
    obs_cuda = {k: v.to("cuda:0") for k, v in obs_cpu.items()}
    dest = OUT / args.label
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "layer_compare.json").exists():
        raise FileExistsError(dest)
    # No hook-free baseline vs hook trace ambiguity: these method wrappers do
    # not populate _forward_hooks and the fused entrypoint is counted below.
    with torch.no_grad():
        baseline_cpu = cpu.logits(obs_cpu)
        baseline_cuda = cuda.logits(obs_cuda).cpu()
    traces = {"cpu": trace(cpu, obs_cpu), "cuda": trace(cuda, obs_cuda)}
    comparison = []
    for a,b in zip(traces["cpu"].events, traces["cuda"].events, strict=True):
        assert a["key"] == b["key"]
        x, y = traces["cpu"].values[a["key"]], traces["cuda"].values[b["key"]]
        finite = torch.isfinite(x) & torch.isfinite(y)
        numeric = x.dtype not in (torch.bool, torch.int64)
        difference = (x[finite] - y[finite]).abs() if numeric else (x[finite] != y[finite]).float()
        comparison.append({"key":a["key"],"name":a["name"],"cpu":a,"cuda":b,"max_abs_common_finite":float(difference.max()) if difference.numel() else None,"finite_pattern_equal":bool(torch.equal(torch.isfinite(x),torch.isfinite(y)))})
    def bad(events):
        return [e for e in events if not e["finite"] and "mask" not in e["name"]]
    summary = {"artifact":str(args.artifact),"model_hash":state_hash(cpu),"mode":"eval/no_grad/float32/autocast_disabled", "backend":backend_info(),"baseline":{"cpu":tensor_info(baseline_cpu),"cuda":tensor_info(baseline_cuda)},"first_bad_cpu":bad(traces["cpu"].events)[:1],"first_bad_cuda":bad(traces["cuda"].events)[:1],"first_cpu_finite_cuda_nonfinite":[r for r in comparison if r["cpu"]["finite"] and not r["cuda"]["finite"] and "mask" not in r["name"]][:1],"layers":comparison}
    for name, baseline in (("cpu",baseline_cpu),("cuda",baseline_cuda)):
        last = next(v for k,v in traces[name].values.items() if k.endswith(":actor.logits"))
        good = torch.isfinite(last) & torch.isfinite(baseline)
        summary[name+"_traced_vs_untraced"]={"finite_pattern_equal":bool(torch.equal(torch.isfinite(last),torch.isfinite(baseline))),"max_abs_common_finite":float((last[good]-baseline[good]).abs().max()) if good.any() else None, "fused_layer_call_count": sum(e["name"] == "fused_transformer.output" for e in traces[name].events)}
    write_json(dest / "layer_compare.json",summary)
    probes={}
    for name, actor, obs in [("cpu",cpu,obs_cpu),("cuda",cuda,obs_cuda)]:
        padding = next(v for k,v in traces[name].values.items() if k.endswith(":encoder.transformer.layers.0.padding_mask"))
        probes[name] = attention_probe(actor,obs,padding)
    summary["implementation"] = args.implementation
    if args.implementation == "current":
        original = original_module.SharedLocalActor(original_module.SharedLocalACNetworkConfig(**payload["network_config"])).eval()
        original.load_state_dict(payload["state_dict"])
        with torch.no_grad():
            old = original.logits(obs_cpu)
            valid = obs_cpu["masks"].any(1)
            summary["valid_rows_vs_original_cpu"] = {"rows": int(valid.sum()), "exact_equal": bool(torch.equal(old[valid], baseline_cpu[valid])), "max_abs": float((old[valid]-baseline_cpu[valid]).abs().max())}
            original = original.to("cuda:0")
            old = original.logits(obs_cuda).cpu()
            summary["valid_rows_vs_original_cuda"] = {"rows": int(valid.sum()), "exact_equal": bool(torch.equal(old[valid], baseline_cuda[valid])), "max_abs": float((old[valid]-baseline_cuda[valid]).abs().max())}
    write_json(dest / "layer_compare.json",summary)
    write_json(dest / "attention_decomposition.json",probes)
    repeat_result = repeats(cuda,obs_cuda,payload["manifest"]["bad_rows"][0],[1,10,100,1000])
    write_json(dest / "minimal_reproducer_result.json", {"artifact":str(args.artifact),"model_hash":state_hash(cpu),"backend":backend_info(),"results":repeat_result})
    print({k:v for k,v in summary.items() if k!='layers'},flush=True)
    print(repeat_result,flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--artifact",type=Path,required=True)
    p.add_argument("--label",required=True)
    p.add_argument("--implementation",choices=["original","current"],default="current")
    run(p.parse_args())
