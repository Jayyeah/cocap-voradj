#!/usr/bin/env python3
"""Canonical BC -> fresh Direct PPO, 5k operational screen, hard 10k stop."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from tools.forward_final_overnight_20260914 import run
if __name__=='__main__':
    run('bc_ppo')
