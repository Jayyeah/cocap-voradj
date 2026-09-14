#!/usr/bin/env python3
"""Single canonical seed, teacher-free, 100k maximum EXPLORATORY_NON_GATE."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from tools.forward_final_overnight_20260914 import run
if __name__=='__main__':
    run('scratch')
