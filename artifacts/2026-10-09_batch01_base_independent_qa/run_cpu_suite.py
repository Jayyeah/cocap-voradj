"""QA-only fixture adapter: use the committed lock on the independent branch.

Production create_lock is unchanged. Four upstream tests assume a Core branch;
this adapter replaces only their lock-generation fixture, never runtime code.
"""
import copy
import json
from pathlib import Path
import pytest
from terl_mappo.batch01 import contracts

HERE = Path(__file__).resolve().parent
lock = json.loads((contracts.ROOT / contracts.LOCK_PATH).read_text())
assert contracts.git('branch', '--show-current') == 'audit/terl-mappo-batch01-base-qa-20261009'
assert contracts.fingerprint(lock) == '91dc6ea76c0f7b76ffa43441286db40b9e4ea1e9e19885a328df16ab6d024bf3'
assert contracts.source_inventory() == lock['common_sources']

def committed_lock_fixture():
    return copy.deepcopy(lock)

contracts.create_lock = committed_lock_fixture
raise SystemExit(pytest.main([
    '-q', '-p', 'no:cacheprovider',
    'test/test_terl_native_mappo_20261008.py',
    'test/test_small_step_ac_migration_contract.py',
    'test/test_mappo_terminal_rows_20260923.py',
    'test/test_terl_mappo_batch01_base_20261009.py',
    '--basetemp=runs/a3_qa02/committed_lock_suite',
    '--junitxml=' + str(HERE / 'committed_lock_suite.xml'),
]))
