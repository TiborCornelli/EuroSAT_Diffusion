# Score-Based Generative Model on EuroSAT

Minimal implementation of a score-based diffusion model trained on the EuroSAT dataset.

## Data

* Dataset: EuroSAT
* Source: Helber et al., *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification*
* Images resized to `64×64`, normalized per-channel

## Forward Process

We use a continuous-time Ornstein–Uhlenbeck type diffusion:

$$
x_t = e^{-t} x_0 + \sqrt{1 - e^{-2t}} ,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

## Score Matching Objective

The network $ s_\theta(x_t, t)$ is trained via denoising score matching:

$$
\mathcal{L} = \mathbb{E}*{t, x_0, \epsilon} \left[
\left| \sigma_t s*\theta(x_t, t) + \epsilon \right|^2
\right]
$$

with

$$
\sigma_t = \sqrt{1 - e^{-2t}}
$$

## Model

* U-Net style architecture
* Residual blocks + GroupNorm + SiLU
* Time embedding via MLP

## Sampling (ULA)

Unadjusted Langevin Algorithm:

$$
x_{k-1} = x_k + h , s_\theta(x_k, t_k) + \sqrt{2h} , \xi_k
$$

with $ \xi_k \sim \mathcal{N}(0, I) $

## Training

* Optimizer: Adam
* LR schedule: Cosine annealing
* Gradient clipping: ( |\nabla| \leq 1 )

## Output

* `loss_history.png`: training curve
* `examples.png`: generated samples
* `score_net_eurosat.pth`: trained weights

## Notes

* Continuous-time formulation (no discrete noise schedule)
* Sampling is slow but simple (no predictor-corrector tricks)
