"""From per-note predictions to tracks ("cluster, then name").

1. Tracklets: in every window, average-linkage clusters of the notes on (1 - track-embedding cosine) + lam (1 - instrument
   distribution cosine), cut at tau; the clusters are chained across overlapping windows by their shared notes (Hungarian
   matching on the overlap); every note keeps the cluster of the window it sits most centrally in.
2. The tracklets of the whole song are merged by size-weighted average linkage on their mean embeddings and instrument
   distributions until the closest pair is farther than tau2.
3. `refine` k-means passes (every note to the nearest centre of a track with >= min_notes notes); tracks still under
   min_notes are folded into the big track with the most similar instrument distribution.
4. Distinct tracks: a track with less than `absorb_frac` of the song's notes survives only if it is clearly its own instrument;
   otherwise its notes are handed, one by one, to the nearest larger track.
5. Naming: alpha log p(track head) + (1 - alpha) log (summed note probabilities); the name maximises the expected credit
   (1 = right instrument, 0.5 = same family). With `merge`, tracks that get the same name are one track.
"""
from __future__ import annotations

import numpy as np

from .instruments import CREDIT


def _cluster(e, p, tau, lam):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    m = len(e)
    if m == 1:
        return np.zeros(1, np.int64)
    e = e.astype(np.float32)
    Dm = 1 - e @ e.T
    if lam > 0:
        pn = p.astype(np.float32)
        pn = pn / np.maximum(np.linalg.norm(pn, axis=1, keepdims=True), 1e-9)
        Dm = Dm + lam * (1 - pn @ pn.T)
    Dm = np.clip((Dm + Dm.T) / 2, 0, None)
    np.fill_diagonal(Dm, 0)
    return fcluster(linkage(squareform(Dm, checks=False), "average"), tau, "distance") - 1


def group_chain(P, on, tau, lam):
    """Per-window clusters chained across windows by shared notes -> tracklet id per note (-1 = none)."""
    from scipy.optimize import linear_sum_assignment
    n = len(on)
    track = -np.ones(n, np.int64)
    best_w = np.full(n, -1.0)
    owner = -np.ones(n, np.int64)
    n_glob = 0
    W = P["W"]
    for t0, sel, e, p in sorted(P["wins"], key=lambda w: w[0]):
        if len(sel) == 0:
            continue
        lab = _cluster(e, p, tau, lam)
        nl = int(lab.max()) + 1
        glob_of = -np.ones(nl, np.int64)
        prev = owner[sel]
        if (prev >= 0).any():
            gids = np.unique(prev[prev >= 0])
            ov = np.zeros((nl, len(gids)))
            np.add.at(ov, (lab[prev >= 0], np.searchsorted(gids, prev[prev >= 0])), 1)
            r, c = linear_sum_assignment(-ov)
            for a_, b_ in zip(r, c):
                if ov[a_, b_] > 0:
                    glob_of[a_] = gids[b_]
        for li in range(nl):
            if glob_of[li] < 0:
                glob_of[li] = n_glob
                n_glob += 1
        g = glob_of[lab]
        owner[sel] = g
        w = np.clip(1.0 - np.abs((on[sel] - t0) / W - 0.5) * 2, 0.05, 1.0)
        better = w > best_w[sel]
        track[sel[better]] = g[better]
        best_w[sel[better]] = w[better]
    return track


def _agglo(spaces, size, tau2, cap=600):
    """Size-weighted average linkage of M groups. spaces = [(means (M, D), weight)]: the distance of two groups is
    sum_w w (1 - m_A . m_B), with m the size-weighted mean of their members' unit vectors (= the mean pairwise cosine distance
    over all member pairs). Merges until the closest pair is farther than tau2 -> a label per group."""
    spaces = [(m.astype(np.float64).copy(), w) for m, w in spaces if w > 0]
    size = size.astype(np.float64).copy()
    M = len(size)
    lab = np.arange(M)
    if M < 2:
        return lab
    Dm = sum(w * (1 - m @ m.T) for m, w in spaces)
    np.fill_diagonal(Dm, np.inf)
    alive = np.ones(M, bool)
    if M > cap:                                                   # very long pieces: only the largest groups merge freely
        small = np.argsort(-size)[cap:]
        Dm[small] = np.inf
        Dm[:, small] = np.inf
    while True:
        k = int(np.argmin(Dm))
        i, j = divmod(k, M)
        if not np.isfinite(Dm[i, j]) or Dm[i, j] > tau2:
            break
        si, sj = size[i], size[j]
        for m, _ in spaces:
            m[i] = (si * m[i] + sj * m[j]) / (si + sj)
        size[i] = si + sj
        alive[j] = False
        lab[lab == j] = i
        Dm[j] = np.inf
        Dm[:, j] = np.inf
        d = sum(w * (1 - m @ m[i]) for m, w in spaces)
        d[~alive] = np.inf
        d[i] = np.inf
        Dm[i] = d
        Dm[:, i] = d
    return lab


def group_tracklet(P, on, tau1, tau2, lam):
    """Tracklets (step 1), then the whole song's tracklets merged (step 2) -> track id per note."""
    tr = group_chain(P, on, tau1, lam)
    idx = np.nonzero(tr >= 0)[0]
    if len(idx) == 0:
        return tr
    tids, inv = np.unique(tr[idx], return_inverse=True)
    M = len(tids)
    if M == 1:
        return tr
    E_, Pm = P["emb"][idx].astype(np.float64), P["probs"][idx].astype(np.float64)
    Pm = Pm / np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-9)
    size = np.bincount(inv, minlength=M).astype(np.float64)
    me = np.zeros((M, E_.shape[1]))
    np.add.at(me, inv, E_)
    me /= size[:, None]
    mp = np.zeros((M, Pm.shape[1]))
    np.add.at(mp, inv, Pm)
    mp /= size[:, None]
    lab = _agglo([(me, 1.0), (mp, lam)], size, tau2)
    out = -np.ones(len(tr), np.int64)
    out[idx] = lab[inv]
    return out


def track_stats(on, off, pitch):
    p = pitch / 64.0 - 1.0
    n = len(p)
    if n <= 3000:
        ov = np.minimum(off[:, None], off[None, :]) > np.maximum(on[:, None], on[None, :]) + 0.03
        poly = float(np.mean(ov.sum(1) > 1.5))
    else:
        poly = 0.0
    return np.array([p.mean(), p.std(), p.min(), p.max(), np.log1p(n) / 5, poly], np.float64)


def absorb_small(inv, emb, on, off, frac, tau_a):
    """Stray tracks: a track holding less than `frac` of the song's notes survives only if it is clearly its own instrument.
    With s = its share of the notes and d = the smallest distance (1 - mean pairwise cosine) to a LARGER track, it is dissolved
    when d < tau_a (1 - s / frac): the smaller the track, the more distinct it has to be. Its notes then go, one by one, to
    the larger track whose centre each is nearest to. inv = track id per note -> new inv."""
    inv = inv.copy()
    n = len(inv)
    if n == 0 or frac <= 0 or tau_a <= 0:
        return inv
    e = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    while True:
        tids, lab = np.unique(inv, return_inverse=True)
        M = len(tids)
        if M < 2:
            break
        size = np.bincount(lab, minlength=M).astype(np.float64)
        cen = np.zeros((M, e.shape[1]))
        np.add.at(cen, lab, e)
        cen /= size[:, None]
        done = True
        for t in np.argsort(size):
            s = size[t] / n
            if s >= frac:
                break
            big = np.nonzero(size > size[t])[0]
            if len(big) == 0:
                continue
            d = 1.0 - cen[big] @ cen[t]
            thr = np.full(len(big), tau_a * (1.0 - s / frac))
            if (d - thr).min() < 0:
                mine = np.nonzero(lab == t)[0]
                cu = cen[big] / np.maximum(np.linalg.norm(cen[big], axis=1, keepdims=True), 1e-9)
                inv[mine] = tids[big[np.argmax(e[mine] @ cu.T, 1)]]
                done = False
                break                                             # sizes / centres changed: start over
        if done:
            break
    return inv


def track_logits(ids, P, notes, th, alpha):
    """Log instrument distribution of one track: alpha log p(track head) + (1 - alpha) log (summed note probabilities)."""
    from scipy.special import erf
    pn = P["probs"][ids].astype(np.float64).sum(0)
    ln = np.log(np.maximum(pn / pn.sum(), 1e-9))
    if th is None or alpha <= 0 or P["h"] is None:
        return ln
    on, off, pitch = notes
    x = np.concatenate([P["h"][ids].astype(np.float64).mean(0), track_stats(on[ids], np.maximum(off[ids], on[ids] + 0.02), pitch[ids])])
    for li, (Wm, b) in enumerate(th):
        x = Wm @ x + b
        if li < len(th) - 1:
            x = 0.5 * x * (1 + erf(x / np.sqrt(2)))
    lt = x - x.max()
    lt = lt - np.log(np.exp(lt).sum())
    return alpha * lt + (1 - alpha) * ln


def decode(P, notes, th, tau=0.1, tau2=0.4, lam=0.5, min_notes=30, refine=2, alpha=0.5, ec=True, merge=True, absorb=0.3,
           absorb_frac=0.25):
    """P = the song's predictions (Grouper.predict), notes = (onset, offset, pitch) arrays, th = the track head's layers
    [(W, b)] -> [(instrument index, sorted note ids)]."""
    on, off, pitch = notes
    if len(on) == 0:
        return []
    track = group_tracklet(P, on, tau, tau2, lam)
    idx = np.nonzero(track >= 0)[0]
    if len(idx) == 0:
        return []
    tids, inv = np.unique(track[idx], return_inverse=True)
    T = len(tids)
    probs = P["probs"]
    size = np.bincount(inv, minlength=T)
    mn = min_notes
    emb = P["emb"][idx].astype(np.float64)
    for _ in range(refine):                                       # every note to the nearest big-track centre
        keepc = np.nonzero(size >= mn)[0]
        if len(keepc) < 2:
            break
        cen = np.zeros((T, emb.shape[1]))
        np.add.at(cen, inv, emb)
        cen = cen[keepc] / np.maximum(np.linalg.norm(cen[keepc], axis=1, keepdims=True), 1e-9)
        sc = emb @ cen.T
        if lam > 0:
            pm = np.zeros((T, probs.shape[1]))
            np.add.at(pm, inv, probs[idx].astype(np.float64))
            pm = pm[keepc] / np.maximum(np.linalg.norm(pm[keepc], axis=1, keepdims=True), 1e-9)
            pn = probs[idx].astype(np.float64)
            sc = sc + lam * (pn / np.maximum(np.linalg.norm(pn, axis=1, keepdims=True), 1e-9)) @ pm.T
        inv = keepc[np.argmax(sc, 1)]
        size = np.bincount(inv, minlength=T)
    mass = np.zeros((T, probs.shape[1]))
    np.add.at(mass, inv, probs[idx])
    big = np.nonzero(size >= mn)[0]
    if len(big) == 0:
        big = np.array([int(np.argmax(size))])
    isbig = np.zeros(T, bool)
    isbig[big] = True
    bign = mass[big] / np.maximum(np.linalg.norm(mass[big], axis=1, keepdims=True), 1e-9)
    src_v = mass / np.maximum(np.linalg.norm(mass, axis=1, keepdims=True), 1e-9)
    for t in range(T):                                            # fold small tracks into the best-matching big one
        if not isbig[t] and size[t] > 0:
            inv[inv == t] = big[int(np.argmax(bign @ src_v[t]))]
    if absorb > 0 and absorb_frac > 0:
        inv = absorb_small(inv, emb, on[idx], off[idx], absorb_frac, absorb)
    out = {}
    for t in np.unique(inv):
        ids = idx[inv == t]
        z = track_logits(ids, P, notes, th, alpha)
        if ec:
            q = np.exp(z - z.max())
            k = int(np.argmax(CREDIT @ (q / q.sum())))
        else:
            k = int(np.argmax(z))
        if merge and k in out:
            out[k] = np.concatenate([out[k], ids])
        else:
            out[k if merge else (k, int(t))] = ids
    return [((k if merge else k[0]), np.sort(ids)) for k, ids in out.items()]
