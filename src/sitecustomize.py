"""Capture actual local exec objects before project imports (no permissions).

Batch01 refuses modules imported without this startup witness. This records
code objects, not source hashes or pyc headers; SourceGuard verifies them against
locked source bytes before accepting a hook, rollout, optimizer or checkpoint.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RECORDS = {}
sys._batch01_exec_records = _RECORDS
sys._batch01_exec_root = _ROOT

def _capture(event, args):
    if event != 'exec':
        return
    code = args[0]
    filename = getattr(code, 'co_filename', '')
    if filename.startswith(_ROOT + os.sep) and filename.endswith('.py'):
        rows = _RECORDS.setdefault(os.path.abspath(filename), [])
        if not any(item is code for item in rows):
            rows.append(code)

sys.addaudithook(_capture)
_RECORDS[os.path.abspath(__file__)] = [sys._getframe().f_code]
