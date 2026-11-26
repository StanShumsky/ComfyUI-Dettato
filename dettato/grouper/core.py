"""Grouping a song's notes into instrument tracks.

    notes (onset, offset, pitch) + the audio
      -> tuning normalisation (the audio is resampled onto A440 when it sits clearly off it)
      -> the transcriber's maps and internal states on that audio (taken from the one transcriber pass that also fed the evidence
         repairer: the average over every other of its windows), gathered at every note (12 scalars + three 368-d taps)
      -> the grouper network over 24 s windows (every 12 s): per-note instrument probabilities and track embeddings,
         averaged over the windows that hold the note (weighted towards the window centre)
      -> the decode (decode.py): tracks, each named by one of 73 instruments
"""
from __future__ import annotations

import threading
import warnings
from contextlib import contextmanager

import numpy as np
import torch

from .. import hub
from ..transcriber.core import _Average
from . import retune
from .decode import decode
from .instruments import INSTRUMENTS, K, PROGRAM
from .model import NoteNet

SR = 16000
PITCH_LO, N_KEYS = 21, 88
PAD0, PAD1 = 0.5, 1.0             # a window's audio starts PAD0 before it and ends PAD1 after it
SPAN_CAP, STEM_SPAN = 50, 10      # frames averaged for tap 1 (the note, <= 1 s) and tap 2 (the attack, 200 ms)
SLOT_TOL = 0.1                    # s: an input note takes the voice slot of the nearest same-pitch transcriber note within this
DEDUP = 0.03                      # s: same-pitch notes this close inside one track are one note
FEAT_DIM = 368


def _autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda"))


_DETERMINISTIC_LOCK = threading.RLock()


@contextmanager
def _deterministic(device):
    """On CUDA, use PyTorch's deterministic kernels while the grouper runs: its front end accumulates with scatter_add, whose
    default CUDA kernel sums in a varying order, and on some songs those last-bit differences change which track a note joins.
    Costs ~20% of the grouper's own (small) run time; CPU results are unaffected."""
    if not str(device).startswith("cuda"):
        yield
        return
    with _DETERMINISTIC_LOCK:
        prev, prev_warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                yield
        finally:
            torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


# ----------------------------------------------------------------------------- note features from the transcriber
def assign_slots(notes, tnotes):
    """Input notes (N, 3) + the transcriber's own notes (M, 4) [onset, offset, pitch, slot 0 / 1] -> (N, 4) sorted by
    (onset, pitch): end at least 20 ms after the onset, voice slot of the nearest same-pitch transcriber note (else 0)."""
    ov = np.asarray(notes, np.float64).reshape(-1, 3)
    slot = np.zeros(len(ov))
    for p in np.unique(ov[:, 2]):
        a = np.where(ov[:, 2] == p)[0]
        b = np.where(tnotes[:, 2] == p)[0]
        if len(b):
            d = np.abs(ov[a, 0][:, None] - tnotes[b, 0][None, :])
            j = d.argmin(1)
            slot[a] = np.where(d[np.arange(len(a)), j] <= SLOT_TOL, tnotes[b[j], 3], 0.0)
    o = np.lexsort((ov[:, 2], ov[:, 0]))
    return np.stack([ov[o, 0], np.maximum(ov[o, 1], ov[o, 0] + 0.02), ov[o, 2], slot[o]], 1)


@torch.no_grad()
def _meta(P, notes, hop):
    dev = P["onset"].device
    T, L = P["onset"].shape
    N = len(notes)
    if N == 0:
        return np.zeros((0, 12), np.float32)
    nt = torch.from_numpy(notes).to(dev)
    lane = (nt[:, 2].round().long() - PITCH_LO).clamp(0, L - 1)
    t0 = (nt[:, 0] / hop).round().long().clamp(0, T - 1)
    te = torch.maximum((nt[:, 1] / hop).ceil().long(), t0 + 1).clamp(max=T)

    def at(m, t):
        return m[t.clamp(0, T - 1), lane]

    def mx3(m, t):
        return torch.stack([at(m, t - 1), at(m, t), at(m, t + 1)]).amax(0)

    def span_mean(m):
        cs = torch.cat([torch.zeros(1, L, device=dev), m.cumsum(0)])
        return (cs[te, lane] - cs[t0, lane]) / (te - t0).clamp(min=1).float()

    meta = torch.stack([nt[:, 0].float(), nt[:, 1].float(), nt[:, 2].float(), nt[:, 3].float(),
                        mx3(P["onset"], t0), mx3(P["onset2"], t0), at(P["frame"], t0), at(P["frame2"], t0),
                        span_mean(P["frame"]), span_mean(P["frame2"]), mx3(P["offset"], te - 1), mx3(P["offset2"], te - 1)], 1)
    return meta.cpu().numpy().astype(np.float32)


@torch.no_grad()
def gather(P, H, S, notes, hop):
    """Maps P {head: (T, 88)}, hidden state H and stem output S (T, 88, C; on the device or in host memory) at the notes (N, 4) ->
    meta (N, 12) float32: onset, offset, pitch, slot, onset prob of each voice (max over +-1 frame), frame prob of each voice
         at the onset and averaged over the note, offset prob of each voice (max over the last frame +-1);
    feat (N, 3, C) int8 + fsc (N, 3) float16 per-row scales: H at the onset, H over the note (<= 1 s), S over the first 200 ms."""
    dev = P["onset"].device
    T = P["onset"].shape[0]
    N = len(notes)
    Cc = H.shape[2]
    if N == 0:
        return np.zeros((0, 12), np.float32), np.zeros((0, 3, Cc), np.int8), np.zeros((0, 3), np.float16)
    nt = torch.from_numpy(notes).to(dev)
    lane = (nt[:, 2].round().long() - PITCH_LO).clamp(0, H.shape[1] - 1)
    t0 = (nt[:, 0] / hop).round().long().clamp(0, T - 1)
    te = torch.maximum((nt[:, 1] / hop).ceil().long(), t0 + 1).clamp(max=T)
    meta = _meta(P, notes, hop)

    def rows(M, t):                 # M[t, lane] on the device, whichever device M lives on
        return M[t.to(M.device), lane.to(M.device)].to(dev)

    def span(M, cap):
        acc = torch.zeros(N, Cc, device=dev)
        cnt = torch.zeros(N, device=dev)
        lim = torch.minimum(te, t0 + cap)
        for k in range(cap):
            t = t0 + k
            m = (t < lim).float()
            acc += rows(M, t.clamp(max=T - 1)).float() * m[:, None]
            cnt += m
        return acc / cnt.clamp(min=1)[:, None]

    X = torch.stack([rows(H, t0).float(), span(H, SPAN_CAP), span(S, STEM_SPAN)], 1)
    sc = X.abs().amax(-1).clamp(min=1e-6) / 127.0
    feat = (X / sc[..., None]).round().clamp(-127, 127).to(torch.int8).cpu().numpy()
    return meta, feat, sc.cpu().numpy().astype(np.float16)


# ----------------------------------------------------------------------------- windows
def song_windows(on, W, hop, nmax):
    """Windows covering every note (onsets `on`, sorted): a fixed time hop; a window holding more than nmax notes is followed
    by one that starts at its first uncovered note. -> [(t0, note indices)]"""
    out, t0 = [], 0.0
    n = len(on)
    if n == 0:
        return out
    end = float(on[-1])
    while True:
        a, b = np.searchsorted(on, t0), np.searchsorted(on, t0 + W)
        sel = np.arange(a, min(b, a + nmax))
        if len(sel) >= 1:
            out.append((t0, sel))
        if b >= n and (a + nmax >= b):
            break
        if b - a > nmax:
            t0 = max(t0 + 0.25 * hop, float(on[a + nmax // 2]))
        else:
            t0 += hop
        if t0 > end:
            break
    return out


class _Song:
    """One song's model inputs: 16-bit audio, note scalars, taps."""

    def __init__(self, y, meta, feat, fsc):
        self.audio = np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.int16)
        self.meta, self.feat, self.fsc = meta, feat, fsc

    def audio_slice(self, t_a, L):
        x = np.zeros(L, np.float32)
        s0 = int(round(t_a * SR))
        lo, hi = max(s0, 0), min(s0 + L, len(self.audio))
        if hi > lo:
            x[lo - s0:hi - s0] = np.asarray(self.audio[lo:hi], np.float32) / 32768.0
        return x

    def window(self, t0, W, sel):
        """One window's inputs, or None if it holds fewer than 4 notes."""
        if len(sel) < 4:
            return None
        notes = self.meta[sel].copy()
        notes[:, 0] -= t0
        notes[:, 1] = np.minimum(notes[:, 1] - t0, W + PAD1)
        feat = np.asarray(self.feat[sel], np.float32) * np.asarray(self.fsc[sel], np.float32)[..., None]
        return dict(audio=self.audio_slice(t0 - PAD0, int(round((PAD0 + W + PAD1) * SR))), notes=notes, feat=feat)


def collate(items, nmax, device):
    B = len(items)
    N = max(8, max(len(it["notes"]) for it in items))
    N = min(nmax, int(np.ceil(N / 64) * 64))
    L = len(items[0]["audio"])
    out = dict(audio=np.zeros((B, L), np.float32), notes=np.zeros((B, N, 12), np.float32), mask=np.zeros((B, N), bool),
               feat=np.zeros((B, N, 3, FEAT_DIM), np.float16))
    for b, it in enumerate(items):
        n = min(N, len(it["notes"]))
        out["audio"][b] = it["audio"]
        out["notes"][b, :n] = it["notes"][:n]
        out["notes"][b, n:, 2] = 60.0
        out["mask"][b, :n] = True
        out["feat"][b, :n] = it["feat"][:n]
    return {k: torch.from_numpy(v).to(device) for k, v in out.items()}


# ----------------------------------------------------------------------------- the grouper
class Grouper:
    """Notes + audio -> instrument-named tracks.

    >>> g = Grouper.from_pretrained("cuda")
    >>> tracks, info = g.group(audio16k, notes, transcriber)   # notes (N, 3): onset s, offset s, MIDI pitch
    >>> tracks[0]["instrument"], tracks[0]["program"], tracks[0]["notes"]   # notes (M, 4): onset, offset, pitch, velocity
    """

    def __init__(self, model: NoteNet, cfg: dict, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.window_s = float(cfg["window_s"])
        self.window_hop = float(cfg["window_hop"])
        self.max_notes = int(cfg["max_notes"])
        self.batch = int(cfg.get("batch", 8))
        self.params = dict(cfg["decode"])
        # the track head, applied in float64 by the decode
        self.th = [(l.weight.detach().float().cpu().numpy().astype(np.float64), l.bias.detach().float().cpu().numpy().astype(np.float64))
                   for l in self.model.heads.th if isinstance(l, torch.nn.Linear)]

    @classmethod
    def from_pretrained(cls, device="cpu"):
        cfg = hub.load_config("grouper")
        assert cfg["instruments"] == INSTRUMENTS, "instrument list of the weights does not match the package"
        model = NoteNet(**cfg["model"])
        model.load_state_dict(hub.load_state("grouper"))
        return cls(model, cfg, device)

    # ------------------------------------------------------------------ steps
    def features(self, y, notes, transcriber, check=None):
        """The transcriber's maps / taps on audio y (its own pass, a window every 4 s), gathered at the notes (N, 3) (already on
        y's time axis). ``group`` does not call this: it takes the same averages from the shared pass (``features_from_run``)."""
        maps, taps = transcriber.run(y, hop_frac=0.5, batch=4, taps=("hidden", "stem"), pad_batch=True, check=check)
        return self.features_from_run(maps, taps, notes, transcriber)

    @torch.no_grad()
    def features_streaming(self, y, notes, transcriber, batch=2, check=None):
        """Grouper features from a bounded pass that retains taps only at frames touched by input notes."""
        dev = transcriber.device
        W, Fw, T, starts, _pad, g_pad = transcriber._layout(len(y), 0.25)
        sub_step = round(W * 0.5)
        nt_np = assign_slots(notes, np.empty((0, 4), np.float64))
        t0 = np.rint(nt_np[:, 0] / transcriber.hop).astype(np.int64).clip(0, T - 1)
        te = np.maximum(np.ceil(nt_np[:, 1] / transcriber.hop).astype(np.int64), t0 + 1).clip(max=T)
        lane = np.rint(nt_np[:, 2]).astype(np.int64) - PITCH_LO
        lane = lane.clip(0, N_KEYS - 1)

        def pair_index(cap, onset=False):
            span = t0[:, None] + np.arange(cap)
            valid = (span >= 0) & (span < T) & (span < np.minimum(te, t0 + cap)[:, None])
            times = np.concatenate([t0[:, None], span], 1) if onset else span
            if onset:
                valid = np.concatenate([np.ones((len(t0), 1), bool), valid], 1)
            keys = times * N_KEYS + lane[:, None]
            unique, inverse = np.unique(keys[valid], return_inverse=True)
            index = np.full(keys.shape, -1, np.int64)
            index[valid] = inverse
            return unique, index

        hkeys, hidx = pair_index(SPAN_CAP, onset=True)
        skeys, sidx = pair_index(STEM_SPAN)
        C = transcriber.channels
        hsum = torch.zeros(len(hkeys), C, dtype=torch.float16, device=dev)
        ssum = torch.zeros(len(skeys), C, dtype=torch.float16, device=dev)
        window = torch.from_numpy(np.hanning(Fw + 2)[1:-1].astype(np.float32)).to(dev)
        wsum = torch.zeros(T, device=dev)
        for s0 in starts:
            if s0 % sub_step:
                continue
            g0 = transcriber.frame_offset(s0, g_pad)
            a, b = max(0, g0), min(T, g0 + Fw)
            if b > a:
                wsum[a:b] += window[a - g0:b - g0]
        wsum.clamp_(min=1e-6)

        def add_pairs(total, keys, tap, k, a, b, g0, wt):
            lo, hi = np.searchsorted(keys, [a * N_KEYS, b * N_KEYS])
            for q in range(lo, hi, 4096):
                pairs = keys[q:min(q + 4096, hi)]
                rel = pairs // N_KEYS - g0
                lanes = pairs % N_KEYS
                total[q:q + len(pairs)] += (tap[k][:, rel, lanes].T.float() * wt[rel, None]).half()

        acc = _Average(T, dev, ())
        for ss, pr, tp in transcriber.windows(y, hop_frac=0.25, batch=batch, taps=("hidden", "stem"), check=check):
            for k, s0 in enumerate(ss):
                if s0 % sub_step:
                    continue
                g0 = transcriber.frame_offset(s0, g_pad)
                a, b = max(0, g0), min(T, g0 + Fw)
                if b <= a:
                    continue
                wt = window[a - g0:b - g0]
                acc.add(a, b, g0, k, wt, pr, None)
                add_pairs(hsum, hkeys, tp["hidden"], k, a, b, g0, window)
                add_pairs(ssum, skeys, tp["stem"], k, a, b, g0, window)
        maps, _ = acc.finish()
        del acc
        hsum /= wsum[hkeys // N_KEYS].half()[:, None]
        ssum /= wsum[skeys // N_KEYS].half()[:, None]
        tn = transcriber.notes(maps)
        own = np.stack([tn["onset"], tn["offset"], tn["pitch"].astype(np.float64), tn["slot"] - 1.0], 1).reshape(-1, 4)
        nt_np = assign_slots(notes, own)
        P = {name: torch.from_numpy(v).to(dev) for name, v in maps.items()}
        meta = _meta(P, nt_np, transcriber.hop)
        mean_h = torch.empty(len(nt_np), C, device=dev)
        mean_s = torch.empty(len(nt_np), C, device=dev)
        for q in range(0, len(nt_np), 256):
            part = slice(q, q + 256)
            hi_idx = torch.from_numpy(hidx[part, 1:]).to(dev)
            si_idx = torch.from_numpy(sidx[part]).to(dev)
            hv = torch.zeros(hi_idx.shape[0], C, device=dev)
            sv = torch.zeros(si_idx.shape[0], C, device=dev)
            hc = torch.zeros(hi_idx.shape[0], device=dev)
            scount = torch.zeros(si_idx.shape[0], device=dev)
            for j in range(SPAN_CAP):
                idx = hi_idx[:, j]
                hv += hsum[idx.clamp(min=0)].float() * (idx >= 0)[:, None]
                hc += idx >= 0
            for j in range(STEM_SPAN):
                idx = si_idx[:, j]
                sv += ssum[idx.clamp(min=0)].float() * (idx >= 0)[:, None]
                scount += idx >= 0
            mean_h[part] = hv / hc.clamp(min=1)[:, None]
            mean_s[part] = sv / scount.clamp(min=1)[:, None]
        at_h = hsum[torch.from_numpy(hidx[:, 0]).to(dev)].float()
        X = torch.stack([at_h, mean_h, mean_s], 1)
        sc = X.abs().amax(-1).clamp(min=1e-6) / 127.0
        feat = (X / sc[..., None]).round().clamp(-127, 127).to(torch.int8).cpu().numpy()
        return meta, feat, sc.cpu().numpy().astype(np.float16)

    def features_from_run(self, maps, taps, notes, transcriber):
        """The same from a transcriber pass already made: ``maps`` and ``taps`` {"hidden", "stem"} averaged over the windows that
        start every 4 s (the sub-run of ``Transcriber.run(y, hop_frac=0.25, sub_hop_frac=0.5, sub_taps=("hidden", "stem"))``)."""
        tn = transcriber.notes(maps)
        own = np.stack([tn["onset"], tn["offset"], tn["pitch"].astype(np.float64), tn["slot"] - 1.0], 1).reshape(-1, 4)
        nt = assign_slots(notes, own)
        P = {k: torch.from_numpy(v).to(transcriber.device) for k, v in maps.items()}
        out = gather(P, taps["hidden"], taps["stem"], nt, transcriber.hop)
        del P, taps
        return out

    @torch.no_grad()
    def predict(self, song: _Song, check=None):
        """-> per-note predictions: probs (n, K), emb (n, E) unit vectors, h (n, d) float16, wins [(t0, sel, emb, probs)]."""
        W, nmax = self.window_s, self.max_notes
        on_all = song.meta[:, 0]
        n = len(on_all)
        acc, wsum, best = np.zeros((n, K), np.float64), np.zeros(n), np.full(n, -1.0)
        eacc = h = None
        wins = []
        jobs = song_windows(on_all, W, W * self.window_hop, nmax)
        for bi in range(0, len(jobs), self.batch):
            if check is not None:
                check()
            chunk = jobs[bi:bi + self.batch]
            items = [song.window(t0, W, sel) or song.window(t0, W, np.concatenate([sel] * 4)[:4]) for t0, sel in chunk]
            batch = collate(items, nmax, self.device)
            with _autocast(self.device), _deterministic(self.device):
                o = self.model(batch)
            p = torch.softmax(o["inst"].float(), -1).cpu().numpy()
            e = o["emb"].float().cpu().numpy()
            hh = o["h"].float().cpu().numpy()
            for k, (t0, sel) in enumerate(chunk):
                m = len(sel)
                on = on_all[sel]
                w = np.clip(1.0 - np.abs((on - t0) / W - 0.5) * 2, 0.05, 1.0)
                acc[sel] += p[k, :m] * w[:, None]
                wsum[sel] += w
                if eacc is None:
                    eacc = np.zeros((n, e.shape[-1]), np.float64)
                    h = np.zeros((n, hh.shape[-1]), np.float16)
                eacc[sel] += e[k, :m] * w[:, None]
                better = w > best[sel]
                h[sel[better]] = hh[k, :m][better]
                best[sel[better]] = w[better]
                wins.append((float(t0), sel, e[k, :m].astype(np.float16), p[k, :m].astype(np.float16)))
        ws = np.maximum(wsum, 1e-9)
        emb = eacc if eacc is not None else np.zeros((n, 1))
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
        return dict(probs=(acc / ws[:, None]).astype(np.float32), emb=emb.astype(np.float32), h=h, wins=wins, W=W)

    # ------------------------------------------------------------------ all of it
    def group(self, audio16k: np.ndarray, notes: np.ndarray, transcriber, tuning=None, run=None, check=None, streamed_features=None):
        """audio16k: the recording (16 kHz mono float32); notes (N, 3): onset s, offset s, MIDI pitch (any order), in the
        recording's time axis; transcriber: a dettato.transcriber.Transcriber (its maps and internal states are features).

        Without ``run`` the tuning is estimated here and the transcriber runs on the retuned audio (a window every 4 s). To share
        one transcriber pass with the evidence repairer, estimate the tuning first (``retune.ratio``), run the transcriber on the
        retuned audio y, and pass ``tuning`` = that ``retune.ratio`` result and ``run`` = (maps, taps) of the sub-average of that
        pass (``features_from_run``). Then ``audio16k`` is y and ``notes`` are on y's time axis; the tracks come back on the
        original axis (times x r) either way. ``streamed_features`` accepts the output of ``features_streaming`` with ``tuning`` for
        long recordings whose full-song taps would be too large.

        -> (tracks, info). tracks: [{"instrument", "program", "notes": (M, 4) [onset, offset, pitch, velocity 1-127]}], the
        largest first; notes sorted by onset."""
        x = np.asarray(audio16k, np.float32)
        nin = np.array(notes, np.float64).reshape(-1, 3)
        if streamed_features is not None:
            if tuning is None:
                raise ValueError("pass the retune.ratio result with precomputed features")
            delta, strength, shift, r = tuning
            dur = len(x) * r / SR
            y = x
            meta, feat, fsc = streamed_features
        elif run is None:
            delta, strength, shift, r = retune.ratio(x)
            y = retune.resample(x, r) if shift else x
            nin[:, :2] /= r                                        # onto the retuned time axis
            dur = len(x) / SR
            meta, feat, fsc = self.features(y, nin, transcriber, check=check)
        else:
            if tuning is None:
                raise ValueError("pass the retune.ratio result with a precomputed run")
            delta, strength, shift, r = tuning
            dur = len(x) * r / SR
            y = x                                                  # already the retuned audio, notes already on its axis
            meta, feat, fsc = self.features_from_run(*run, nin, transcriber)
        song = _Song(y, meta, feat, fsc)
        on, off, pitch = meta[:, 0].astype(np.float64), meta[:, 1].astype(np.float64), meta[:, 2].round().astype(np.int64)
        conf = np.where(meta[:, 3] > 0.5, meta[:, 5], meta[:, 4])
        groups = decode(self.predict(song, check=check), (on, off, pitch.astype(np.float64)), self.th, **self.params) if len(on) else []
        tracks = self._tracks(on * r, off * r, pitch, conf, groups)
        info = dict(tuning_cents=round(float(delta), 2), tuning_strength=round(float(strength), 3), retune_cents=round(float(shift), 2),
                    retune_ratio=float(r), duration_s=round(dur, 2), notes_in=len(on),
                    notes_out=int(sum(len(t["notes"]) for t in tracks)))
        return tracks, info

    @staticmethod
    def _tracks(on, off, pitch, conf, groups):
        """Decoded groups -> track dicts (same-pitch notes within 30 ms in one track become one note)."""
        out = []
        for k, ids in sorted(groups, key=lambda t: -len(t[1])):
            ids = np.asarray(ids)
            o = ids[np.lexsort((on[ids], pitch[ids]))]
            keep, last = [], None
            for i in o:
                if last is not None and pitch[i] == pitch[last] and on[i] - on[last] <= DEDUP:
                    continue
                keep.append(i)
                last = i
            rows = [(float(on[i]), float(max(off[i], on[i] + 0.02)), int(pitch[i]), int(np.clip(20 + 107 * conf[i], 1, 127)))
                    for i in sorted(keep, key=lambda q: on[q])]
            out.append({"instrument": INSTRUMENTS[k], "program": int(PROGRAM[k]),
                        "notes": np.array(rows, np.float64).reshape(-1, 4), "merged_duplicates": len(o) - len(keep)})
        return out
