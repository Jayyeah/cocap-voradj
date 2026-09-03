#!/usr/bin/env python3
"""Run the authorized IQN-VXY support-reward 1M-per-stage curriculum.

The implementation lives in the audited full-curriculum supervisor.  This
thin entrypoint pins the support11 profile so an accidental default-profile
launch cannot write into the historical run.
"""
from __future__ import annotations

import sys

from tools import supervise_iqn_vxy_full_20260830 as supervisor


def main(argv: list[str] | None = None) -> int:
    return supervisor.main(argv, profile_override="support11")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
