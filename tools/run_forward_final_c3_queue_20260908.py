#!/usr/bin/env python3
"""Run immutable frozen C3 modes sequentially on one already-locked GPU."""
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from tools.run_forward_final_bridge_20260908 import atomic_json
from tools.evaluate_forward_final_actor_20260908 import aggregate,MODES


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--modes',nargs='+',choices=MODES,required=True)
    p.add_argument('--actor-checkpoint',type=Path,required=True);p.add_argument('--physical-gpu',type=int,required=True);args=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(args.physical_gpu)
    source_files=['tools/run_forward_final_bridge_20260908.py','tools/evaluate_forward_final_actor_20260908.py','src/cocap_voradj/training/forward_final.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source_files}
    for mode in args.modes:
        assert hashes=={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source_files},'Evaluator changed during queue'
        out=args.root/mode;out.mkdir(parents=True,exist_ok=False)
        cmd=[sys.executable,str(ROOT/'tools/evaluate_forward_final_actor_20260908.py'),'--output-root',str(out),'--mode',mode,'--actor-checkpoint',str(args.actor_checkpoint),'--episodes','100','--seed','2026096101','--device','cuda:0']
        with (out/'run.log').open('w') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            meta={'stage':'C3 frozen formal100 per scene','mode':mode,'pid':child.pid,'queue_pid':os.getpid(),'started_at':datetime.datetime.now().astimezone().isoformat(),'physical_gpu':args.physical_gpu,'command':cmd,'source_sha256':hashes}
            atomic_json(out/'process.json',meta);atomic_json(args.root/f'queue_gpu{args.physical_gpu}.json',{'status':'running',**meta,'modes':args.modes})
            code=child.wait()
        if code:
            atomic_json(args.root/f'queue_gpu{args.physical_gpu}.json',{'status':'failed','mode':mode,'returncode':code,'pid':child.pid});raise SystemExit(code)
    atomic_json(args.root/f'queue_gpu{args.physical_gpu}.json',{'status':'complete','modes':args.modes,'physical_gpu':args.physical_gpu})
    if all((args.root/m/'report.json').exists() for m in MODES):aggregate(args.root)

if __name__=='__main__':main()
