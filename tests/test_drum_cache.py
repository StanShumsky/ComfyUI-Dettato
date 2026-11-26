import importlib
import threading

import torch


def test_filter_cache_initialization_is_atomic(monkeypatch):
    features = importlib.import_module("dettanode.dettato.drums.features")
    original = features.mel_filterbank
    hi_started = threading.Event()
    second_attempted = threading.Event()
    release = threading.Event()
    calls = 0

    def controlled_mel(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            hi_started.set()
            if not release.wait(2):
                raise AssertionError("timed out releasing filter initialization")
        return original(*args, **kwargs)

    monkeypatch.setattr(features, "mel_filterbank", controlled_mel)
    features._FILTERS.clear()
    results, errors = {}, []

    def worker(name):
        try:
            if name == "hi":
                second_attempted.set()
            results[name] = features._filters(name)
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=worker, args=("lo",))
    second = threading.Thread(target=worker, args=("hi",))
    first.start()
    assert hi_started.wait(1)
    second.start()
    assert second_attempted.wait(1)

    try:
        assert not results
    finally:
        release.set()
        first.join(2)
        second.join(2)

    assert not first.is_alive() and not second.is_alive()
    assert not errors
    assert set(features._FILTERS) == {"lo", "hi", "cq"}
    assert torch.equal(results["lo"], features._filters("lo"))
    assert torch.equal(results["hi"], features._filters("hi"))
