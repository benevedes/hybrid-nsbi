# Hybrid NSBI techniques for robust and inexpensive inference

Code implementing the Latent Categories and Mixture of Summary Statistics methods to reproduce the results and figures of [2609.36044](https://arxiv.org/abs/2609.36044) in the Gaussian covariance and the off-shell Higgs
(gg → ZZ → 4ℓ) case studies.

## Layout

```
latentcat/
  common.py          encoder builder, binned Poisson scan, WeightedEventData (train / val / test split, reweighted templates,
                     exact ceiling, training-event sampling) shared by both domains
  gaussian_cov_data.py  5-d Gaussian covariance toy, simulated at 11 values of θ and reweighted to any θ (balance heuristic)
  physics_data.py    gg -> ZZ -> 4l samples, exact matrix-element reweighting to any mu (balance heuristic)
  training.py        train_one_event, the one-event categorical-likelihood objective
  coverage.py        toy coverage: refined likelihood maxima and the coverage figure
scripts/
  gaussian_train_event.py, gaussian_train_nsbi.py, gaussian_eval.py, gaussian_figure.py, gaussian_occupancy.py,
  gaussian_coverage.py
  physics_cache_csv.py, physics_train_event.py, physics_train_carl.py, physics_eval.py, physics_eval_carl.py,
  physics_figure.py, physics_occupancy.py, physics_coverage.py
results/{gaussian,physics}/   run records (run_*.json), encoder states (state_*.pt), fine-grid evaluations,
                              the Gaussian NSBI ensemble (nsbi.json, state_nsbi.pt), and the two physics CARL networks
                              (carl_{sig,sbi}.pt, ~67 MB each) with their scans (eval101_carl.json), and the
                              pseudo-experiments of the coverage studies (coverage.{json,npz})
figures/                      gaussian_fig3.pdf, physics_fig4.pdf (the scans), gaussian_occupancy.pdf,
                              physics_occupancy.pdf (what the learned binning looks like), gaussian_coverage.pdf,
                              physics_coverage.pdf (observed vs expected coverage of the -2ΔlnL intervals)
run_all.sh                    everything, in order
```

## Setup

```
pip install -r requirements.txt
```

The Gaussian toy needs no inputs (it simulates itself). The physics case needs the files listed in `data/README.md`.
`data/` can be pointed elsewhere with `LATENTCAT_DATA=/path/to/data`.

## Running

```
./run_all.sh
```

Scripts pick CUDA, then Apple MPS, then CPU (override with `--device`). `run_all.sh` runs the training jobs in parallel,
at most `MAX_JOBS` at a time (default 4; each physics job needs about 6 GB of memory). On a multi-GPU node, `NGPUS=n`
spreads the jobs over the GPUs, e.g. `MAX_JOBS=9 NGPUS=4 ./run_all.sh` on one 4-GPU node. The two CARL networks
dominate: about 7 hours each on an A100 (500 epochs); they save a resume point every epoch, so an interrupted run
continues where it stopped. Everything else takes about an hour on the same node. Except for training the CARL networks (used for the NSBI and MSS methods) for the physics case study, everything runs on a laptop in a reasonable amount of time.
Converting the physics CSVs takes a few minutes and is done once.

To rebuild every evaluation and figure from the trained networks committed in `results/`, without retraining:

```
./run_all.sh --figures
```

This takes a few minutes (most of it evaluating the two CARL networks on the physics events; a GPU helps) and needs
the same inputs as a full run.
