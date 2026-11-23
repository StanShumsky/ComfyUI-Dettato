"""Repro against the REAL ComfyUI runtime (no stubs): the user's scenarios."""
import importlib.util, os, sys, traceback

CUI = r"C:\Users\shums\ComfyUI-Installs\ComfyUI\ComfyUI"
import glob
_pack_dirs = glob.glob(os.path.join(CUI, "custom_nodes", "ComfyUI-" + "Dettato"))
assert _pack_dirs, "dettato pack not found in custom_nodes"
PACK = _pack_dirs[0]
print("pack:", PACK, flush=True)
sys.path.insert(0, CUI)
os.chdir(CUI)  # real folder_paths defaults (models/, output/)

import numpy as np
import torch
import comfy.model_management as mm
print("real comfy model_management:", mm.__file__, flush=True)

spec = importlib.util.spec_from_file_location(
    "dhetto_pack", os.path.join(PACK, "__init__.py"), submodule_search_locations=[PACK])
pkg = importlib.util.module_from_spec(spec)
sys.modules["dhetto_pack"] = pkg
spec.loader.exec_module(pkg)
nodes = sys.modules["dhetto_pack.nodes"]


def synth_song(t=4.0, rate=48000):
    freqs = [261.63, 329.63, 392.0, 523.25]
    n = int(t * rate)
    x = np.zeros(n, np.float32)
    for i, f in enumerate(freqs):
        a, b = int(i * t / len(freqs) * rate), int((i + 1) * t / len(freqs) * rate)
        seg = np.arange(b - a)
        env = np.clip(np.minimum(seg / 100.0, (b - a - seg) / 500.0), 0.0, 1.0)
        tone = sum(np.sin(2 * np.pi * f * h * seg / rate) / h for h in (1, 2, 3, 4))
        x[a:b] += 0.25 * env * tone
    return {"waveform": torch.from_numpy(x)[None, None], "sample_rate": rate}


def run_case(precision, device):
    print(f"\n===== precision={precision} device={device} =====", flush=True)
    try:
        w = nodes.DettatoLoader.execute("dettato", precision, device).args[0]
        for st in (nodes.DettatoEvidence, nodes.DettatoGrouper, nodes.DettatoDrums):
            w = st.execute(w).args[0]
        payload, text = nodes.DettatoTranscribe.execute(w, {"audio_0": synth_song()}, 1).args
        n_notes = sum(len(np.asarray(t.notes, np.float64).reshape(-1, 4)) for t in payload["tracks"])
        print(f"OK: {len(payload['tracks'])} tracks, {n_notes} notes, "
              f"drums={str(payload['info'].get('drums'))[:80]}", flush=True)
    except Exception:
        traceback.print_exc()


for prec, dev in [("bf16", "cuda"), ("fp16", "cuda"), ("auto", "cuda")]:
    run_case(prec, dev)
