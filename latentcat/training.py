"""The training loop shared by the Gaussian toy and the physics case. It takes a WeightedEventData `d` (which owns the
training pool, event sampling and val / test evaluation) and an `evaluate(enc)` callback that scores on
the VALIDATION set and returns a dict with at least "q_lat" (Asimov scan of the argmax binning) and "active"; any
other "q_*" entries are logged. The test set is never touched here.

train_one_event: the one-event categorical-likelihood objective. One optimizer step:
    1. templates: a random batch of training events, encoded once; for every grid point the template p_k(theta)
       is the weighted soft occupancy sum_i W[i, theta] s_ik, normalized over k   (gradients flow through)
    2. labelled events: n_events training events, each drawn at a uniformly random grid point, with soft
       assignments s_k
    3. score every grid point by the categorical log-likelihood of the event, L(theta) = sum_k s_k log p_k(theta)
    4. loss = cross-entropy of softmax_theta L(theta) against the true grid index (= -log posterior of theta given
       one categorized event, flat prior)
   Its expectation is log(n_grid) - I(theta; k), so the encoder maximizes the mutual information between theta and one
   event's category, which to leading order is the prior-averaged Asimov -2 dlnL of the binned analysis divided by 2N,
   for any dataset size N. The rate term N log nu - nu of the extended likelihood is left out: the binning cannot
   change the information the total count carries, and at N = 1 the term would swamp the shape term. The rate returns
   in every scan. No set classifier, no entropy penalty.

Checkpoint selection never uses the exact likelihood or the test set: it is d.score of the validation scan, the scan's
own mean -2 dlnL at REPORT_POINTS.
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _log_line(d, n_cat, ev):
    parts = [f"val active={ev['active']}/{n_cat}"] + [
        d.report(k[2:], v, part="val") for k, v in ev.items() if k.startswith("q_")
    ]
    return " | ".join(parts)


def _history_entry(ev, **extra):
    return extra | {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ev.items()}


def _copy_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def train_one_event(d, enc, n_cat, evaluate, epochs, steps, n_events, tmpl_batch, tau0, tau1, lr, weight_decay, eval_every):
    opt = optim.Adam(enc.parameters(), lr=lr, weight_decay=weight_decay)
    # tensors used every step
    tr_t = torch.tensor(d.tr, device=d.device)  # training-event indices
    Wtr_t = torch.tensor(d.Wtr, dtype=torch.float32, device=d.device)  # their weights at every grid point, (N_tr, n_grid)

    best, best_state, history = -np.inf, None, []
    t0 = time.time()
    for ep in range(epochs):
        tau = tau0 * (tau1 / tau0) ** (ep / max(1, epochs - 1))  # geometric anneal tau0 -> tau1 over the run
        losses = []
        for _ in range(steps):
            # 1. templates for all grid points from one encoder pass over a random training batch
            tb = torch.tensor(d.rng.choice(len(d.tr), size=tmpl_batch, replace=False), device=d.device)
            s = torch.softmax(enc(d.Xt[tr_t[tb]]) / tau, -1)  # (T, n_cat) soft assignments
            p = Wtr_t[tb].T @ s  # (n_grid, n_cat) weighted soft occupancy = templates
            p = (p + 1e-6) / (p + 1e-6).sum(-1, keepdim=True)
            # 2. labelled events at uniformly random grid points (one sampling call per grid point, not per event)
            counts = np.bincount(d.rng.integers(d.n_grid, size=n_events), minlength=d.n_grid)
            ev_idx = np.concatenate([d.sample_events(j, c) for j, c in enumerate(counts)])
            y = torch.tensor(np.repeat(np.arange(d.n_grid), counts), device=d.device)
            n = torch.softmax(enc(d.Xt[torch.tensor(ev_idx, device=d.device)]) / tau, -1)  # (n_events, n_cat)
            # 3. categorical log-likelihood of every event at every grid point
            L = n @ torch.log(p).T  # (n_events, n_grid)
            # 4. -log posterior of the true grid point
            loss = nn.functional.cross_entropy(L, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        line = f"epoch {ep:3d} tau={tau:.3f} loss={np.mean(losses):.5f} [{(time.time()-t0)/60:.1f} min]"
        if ep % eval_every == 0 or ep == epochs - 1:
            ev = evaluate(enc)
            score = d.score(ev["q_lat"])
            history.append(_history_entry(ev, epoch=ep, tau=tau, loss=float(np.mean(losses)), score=score))
            line += " | " + _log_line(d, n_cat, ev)
            if score > best:
                best, best_state = score, _copy_state(enc)
        print(line, flush=True)
    return best_state, history
