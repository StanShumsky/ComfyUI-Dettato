import threading

from dettanode.dettato.grouper.core import _deterministic


def test_deterministic_contexts_do_not_overlap(monkeypatch):
    state = dict(enabled=False, warn_only=False, active=0, max_active=0)
    entered = [threading.Event(), threading.Event()]
    release = threading.Event()

    def use_deterministic(enabled, warn_only=False):
        state["enabled"] = enabled
        state["warn_only"] = warn_only

    monkeypatch.setattr("dettanode.dettato.grouper.core.torch.are_deterministic_algorithms_enabled", lambda: state["enabled"])
    monkeypatch.setattr("dettanode.dettato.grouper.core.torch.is_deterministic_algorithms_warn_only_enabled", lambda: state["warn_only"])
    monkeypatch.setattr("dettanode.dettato.grouper.core.torch.use_deterministic_algorithms", use_deterministic)

    def worker(i):
        with _deterministic("cuda"):
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            entered[i].set()
            if not release.wait(2):
                raise AssertionError("timed out waiting to release deterministic context")
            state["active"] -= 1

    first = threading.Thread(target=worker, args=(0,))
    second = threading.Thread(target=worker, args=(1,))
    first.start()
    assert entered[0].wait(1)
    second.start()

    try:
        assert not entered[1].wait(0.1)
    finally:
        release.set()
        first.join(2)
        second.join(2)

    assert not first.is_alive() and not second.is_alive()
    assert state["max_active"] == 1
    assert state["enabled"] is False
    assert state["warn_only"] is False
