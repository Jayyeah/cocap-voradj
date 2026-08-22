import os
import signal

from maadpg_reproduction.cli import _install_signal_handlers


def test_sigterm_sets_deferred_request_without_raising_mid_transition():
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)
    try:
        request = _install_signal_handlers()
        os.kill(os.getpid(), signal.SIGTERM)
        assert request["reason"] == f"received signal {signal.SIGTERM}"
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
