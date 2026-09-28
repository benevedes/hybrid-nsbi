"""
One CARL ratio network for the gg -> ZZ -> 4l case: a classifier between one process (signal or SBI) and background,
whose output gives the density ratio r_J(x) = p_J(x) / p_bkg(x). The NSBI and MSS curves of the physics figure are built
from the two networks (--proc sig and --proc sbi); see scripts/physics_eval_carl.py.

The setup uses a single network per process:
  * MLP (latentcat.common.carl_mlp): Linear(17, 1024) + SiLU, then 16 x [Linear(1024, 1024) + SiLU], then
    Linear(1024, 1) + Sigmoid; Xavier-uniform weights, zero biases;
  * weighted binary cross-entropy on the sigmoid output, loss = sum(w * bce) / sum(w) per batch, with the weights of
    each class summing to 1; process J has label 1, background label 0;
  * NAdam at lr 1e-4 (at 1e-3 a single large update saturates the float32 sigmoid within the first ~150 steps, after
    which the clamped BCE gives an exactly zero gradient); ReduceLROnPlateau (factor 0.1, patience 5 epochs) on the
    validation loss; early stopping after 20 epochs without a lower validation loss, which in practice does not
    trigger (the loss keeps decreasing slightly), so training runs to the 500-epoch limit; batch 1024; the training
    set is shuffled once and read in the same order every epoch;
  * model selection: of the 5 checkpoints with the lowest validation loss, the one from the latest epoch;
  * no ensembling, and no calibration or normalization after training: r_J = s / (1 - s) = exp(logit).

Data: every event of the training pool enters both classes, weighted by its exact matrix-element weight for process J
and for background (the reweighted simulation of latentcat.physics_data); the validation part is used the same way, and
the test part is never touched. Process weights: w_bkg = W(mu=0), w_sbi = W(mu=1), w_sig = (W(4) - 2 W(1) + W(0)) / 2.
Features: the 17 of latentcat.physics_data.FEATS, standardized on the training pool.

Writes results/physics/carl_<proc><_tag>.pt (the selected state, its epoch, the loss history, args). A resume file
carl_<proc><_tag>_resume.pt (not committed) is written after every epoch, so an interrupted job continues where it
stopped when rerun. Cost: about 0.8 min per epoch on an A100, roughly 7 hours per network; far longer without a GPU.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latentcat.common import carl_mlp, default_device, results_dir
from latentcat.physics_data import FEATS, PhysicsData

ap = argparse.ArgumentParser()
ap.add_argument("--proc", required=True, choices=["sig", "sbi"])
ap.add_argument("--out", default=None, help="default results/physics")
ap.add_argument("--tag", default="")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--n_layers", type=int, default=16)
ap.add_argument("--n_nodes", type=int, default=1024)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--max_epochs", type=int, default=500)
ap.add_argument("--lr_patience", type=int, default=5)
ap.add_argument("--stop_patience", type=int, default=20)
ap.add_argument("--top_k", type=int, default=5)
ap.add_argument("--device", default=default_device())
a = ap.parse_args()

torch.set_float32_matmul_precision("high")  # TF32 on CUDA
OUT = a.out or results_dir("physics")
os.makedirs(OUT, exist_ok=True)
SFX = f"{a.proc}_{a.tag}" if a.tag else a.proc
F_STATE, F_RESUME = (os.path.join(OUT, f) for f in (f"carl_{SFX}.pt", f"carl_{SFX}_resume.pt"))
dev = torch.device(a.device)

d = PhysicsData(n_mu=21, device="cpu")
j0, j1, j4 = (int(np.argmin(np.abs(d.GRID - m))) for m in (0.0, 1.0, 4.0))
w_all = {"bkg": d.W[:, j0], "sbi": d.W[:, j1], "sig": np.clip(0.5 * (d.W[:, j4] - 2 * d.W[:, j1] + d.W[:, j0]), 0, None)}
X_all = d.Xt.to(dev)


def balanced(pool, rng):
    """(event index, label, weight) tensors: every event of `pool` as numerator (w_proc) and denominator (w_bkg),
    each class normalized to sum 1, zero-weight entries dropped, shuffled once."""
    idx, lab, w = [], [], []
    for k, y in ((a.proc, 1.0), ("bkg", 0.0)):
        wk = w_all[k][pool]
        keep = wk > 0
        idx.append(pool[keep])
        lab.append(np.full(keep.sum(), y, np.float32))
        w.append((wk[keep] / wk.sum()).astype(np.float32))
    idx, lab, w = np.concatenate(idx), np.concatenate(lab), np.concatenate(w)
    p = rng.permutation(len(idx))
    return (torch.tensor(v[p], device=dev) for v in (idx, lab, w))


rng = np.random.default_rng(a.seed)
it, yt, wt = balanced(d.tr, rng)
iv, yv, wv = balanced(d.idx["val"], rng)
print(f"{a.proc}: {len(yt)} training / {len(yv)} validation entries, {len(FEATS)} features, device {dev}", flush=True)

torch.manual_seed(a.seed)
body = carl_mlp(len(FEATS), a.n_nodes, a.n_layers).to(dev)  # pre-sigmoid output; the Sigmoid is applied in the loss


def xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


body.apply(xavier)
bce = nn.BCELoss(reduction="none")  # on sigmoid outputs (BCELoss clamps the log at -100)
opt = torch.optim.NAdam(body.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=a.lr_patience)


def batch_loss(idx, y, w):
    return (bce(torch.sigmoid(body(X_all[idx]).flatten()), y) * w).sum() / w.sum()


@torch.no_grad()
def val_loss(B=16384):
    """Epoch validation loss, as Lightning logs it: the mean of per-batch losses weighted by batch size."""
    body.eval()
    tot = sum(batch_loss(iv[s : s + B], yv[s : s + B], wv[s : s + B]).item() * len(yv[s : s + B]) for s in range(0, len(yv), B))
    body.train()
    return tot / len(yv)


cpu_state = lambda: {k: v.detach().cpu().clone() for k, v in body.state_dict().items()}
hist, topk, best, since_best, ep0 = [], [], np.inf, 0, 0  # topk: [(val_loss, epoch, state)]
if os.path.exists(F_RESUME):
    R = torch.load(F_RESUME, map_location="cpu", weights_only=False)
    body.load_state_dict(R["model"])
    opt.load_state_dict(R["opt"])
    sched.load_state_dict(R["sched"])
    hist, topk, best, since_best, ep0 = R["hist"], R["topk"], R["best"], R["since_best"], R["epoch"] + 1
    torch.set_rng_state(R["torch_rng"])
    print(f"resumed after epoch {R['epoch']}, best val {best:.6f}", flush=True)

t0 = time.time()
for ep in range(ep0, a.max_epochs):
    if since_best >= a.stop_patience:
        break
    tr_sum = 0.0
    for s in range(0, len(yt), a.batch):
        loss = batch_loss(it[s : s + a.batch], yt[s : s + a.batch], wt[s : s + a.batch])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tr_sum += loss.detach() * len(yt[s : s + a.batch])  # no per-step sync
    vl = val_loss()
    lr = opt.param_groups[0]["lr"]
    sched.step(vl)
    hist.append(dict(epoch=ep, train_loss=float(tr_sum) / len(yt), val_loss=vl, lr=lr))
    if vl < best:
        best, since_best = vl, 0
    else:
        since_best += 1
    if len(topk) < a.top_k or vl < max(t[0] for t in topk):
        topk.append((vl, ep, cpu_state()))
        topk = sorted(topk, key=lambda t: t[0])[: a.top_k]
    print(f"epoch {ep:3d}  train {hist[-1]['train_loss']:.6f}  val {vl:.6f}  lr {lr:.0e}  best {best:.6f}  "
          f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
    torch.save(dict(model=body.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), hist=hist, topk=topk, best=best,
                    since_best=since_best, epoch=ep, torch_rng=torch.get_rng_state()), F_RESUME + ".tmp")
    os.replace(F_RESUME + ".tmp", F_RESUME)

final = max(topk, key=lambda t: t[1])  # the latest epoch among the top-k validation checkpoints
print(f"stopped after epoch {hist[-1]['epoch']}: using epoch {final[1]} (val {final[0]:.6f})", flush=True)
torch.save(dict(state=final[2], epoch=final[1], hist=hist, args=vars(a), mean=d.mean, std=d.std, feats=FEATS), F_STATE)
print(f"saved {F_STATE}", flush=True)
