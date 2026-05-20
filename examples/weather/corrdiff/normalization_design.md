# Output normalization for CorrDiff regression on Europa

This document analyses how output channels are normalized in the CorrDiff
regression UNet, explains why the CWA-tuned defaults misbehave on
European precipitation, and recommends a variance-stabilizing transform
as the scientifically appropriate fix. The argument is written at
thesis-grade detail because the choice materially affects the loss the
regression model optimises.

---

## 1. TL;DR

* CorrDiff trains the regression stage with an unweighted MSE in
  *normalized* space across all four high-res output channels.
* On the Europa store (`cwb_scale = [8.24, 3.43, 3.26, 0.667]` for T2,
  U10, V10, precip_1hr), a **1 mm** error in hourly precipitation
  contributes **~152×** more to the loss than a 1 K error in 2 m
  temperature under the default v1 normalization. This is an artefact
  of z-scoring a zero-inflated, heavy-tailed variable, not an
  intentional re-weighting.
* CWA's `v2` normalization handled the analogous problem for radar
  reflectivity with hand-tuned constants (`center=25, scale=25`). That
  branch is **silently inactive** on Europa because it matches the
  variable name `"maximum_radar_reflectivity"`, while Europa uses
  `"precipitation_amount_1hr"`.
* **Implemented (2026-05-20): a Europe-tuned linear normalization**
  (`normalization: europa`) — anchors winds and precipitation at their
  natural zero, picks `scale=5 mm` for precipitation, keeps the
  empirical wind/T2 stats. This brings precip's implicit MSE weight
  from 152× T2 down to 2.7× T2 with **zero non-linear transforms**.
  Function: `get_target_normalizations_europa` in
  [cwb.py:57-85](datasets/cwb.py#L57-L85). Selected from the dataset
  config via `normalization: europa`.
* **Future refinement (not implemented):** apply a **log1p
  variance-stabilizing transform** to the precipitation channel before
  z-scoring. This is the standard practice in the modern
  precipitation-ML literature and additionally addresses the
  heavy-tail training-stability concern (a single 50 mm/h convective
  pixel dominating the minibatch gradient under linear scaling). The
  current implementation does *not* do this — escalate to log1p only
  if the linear-rescale baseline shows loss-spike instability during
  training. Implementation sketch in §8 below.
* Bias caveat (only relevant if log1p is later adopted): log1p is
  non-linear, so back-transforming the regression output via `expm1`
  introduces a small negative bias on the conditional mean (Jensen's
  inequality). In the full two-stage CorrDiff pipeline the
  residual-diffusion stage absorbs this bias.

---

## 2. How the loss reaches the model

The regression UNet outputs four channels in *normalized* space. The
loss is the standard L2:

$$
\mathcal{L}_\mathrm{reg} \;=\; \mathbb{E}_{\mathrm{pixel}, c}\;\Big( \hat{z}_c - z_c \Big)^2,
\qquad z_c = \frac{y_c - \mu_c}{\sigma_c}
$$

where $y_c$ is the physical value in channel $c$, and $(\mu_c, \sigma_c)$
are the center/scale read from the zarr (the function
`get_target_normalizations_v1` in [`datasets/cwb.py`](datasets/cwb.py#L36-L38)).

For a fixed physical error $\Delta y_c$ in channel $c$, the
contribution to the per-pixel MSE is:

$$
\Delta \mathcal{L}_c = \frac{(\Delta y_c)^2}{\sigma_c^2}.
$$

Equal MSE contribution per channel therefore requires the implicit
weight $w_c = 1/\sigma_c^2$ to match the relative importance you assign
to a unit physical error in channel $c$. The choice of $\sigma_c$ is
*the* hyperparameter that decides cross-channel balance.

---

## 3. The CWA defaults (`v1`, `v2`)

`v1` ([`cwb.py:36-38`](datasets/cwb.py#L36-L38)) is plain z-scoring with
empirical mean and std read from the store. `v2`
([`cwb.py:41-54`](datasets/cwb.py#L41-L54)) keeps `v1` for "Gaussian"
channels and **overrides** the non-Gaussian ones by channel-name match:

| Channel | v1 (Europa stats) | v2 override |
|---|---|---|
| `temperature_2m`            | μ≈283.24, σ≈8.24 | (no override) |
| `eastward_wind_10m`         | μ≈0.76,   σ≈3.43 | μ=0,  σ=20 |
| `northward_wind_10m`        | μ≈0.31,   σ≈3.26 | μ=0,  σ=20 |
| `maximum_radar_reflectivity`| —                | μ=25, σ=25 |
| `precipitation_amount_1hr`  | μ≈0.108,  σ≈0.667| **— (no override)** |

Two design ideas are embedded in `v2`:

1. **Anchor wind components at zero**, not at the empirical mean. Wind
   has a physically meaningful zero and is symmetric in magnitude; the
   empirical mean only captures the prevailing synoptic flow over the
   training period, which is a domain-specific bias we don't want
   baked into the target distribution.
2. **Hand-pick a `σ` that produces a usable dynamic range** for
   channels whose distribution is heavy-tailed or strongly non-Gaussian.
   For radar dBZ (a bounded but skewed quantity in roughly [0, 75]),
   `center=25, scale=25` maps the bulk of the distribution into
   $[-1, 2]$, taming the tail without a non-linear transform.

`v2` is therefore *not* a generic recipe; it is a hand-tuned override
table for CWA's specific channel set.

---

## 4. Why `v1` misbehaves on Europa precip

The Europa precipitation channel is fundamentally different from any
CWA channel:

* **Bounded below by zero**, with no upper bound (a "censored on the
  left" distribution).
* **Zero-inflated** — at hourly resolution over central Europe a large
  majority of pixels in any one timestep are 0 mm. The Würzburg-domain
  stats give μ≈0.108 mm, σ≈0.667 mm; both are dominated by the long
  zero-mass.
* **Heavy-tailed when conditional on nonzero precip** — when it does
  rain, the distribution of $\log(\mathrm{precip}\mid\mathrm{precip}>0)$
  is roughly Gaussian (a classical empirical fact). Rare events of
  10-50 mm/h occur and matter.
* **Multiplicative in nature** — half a millimetre and five millimetres
  are *qualitatively* different events; doubling precipitation has a
  similar physical meaning across the range. This is unlike
  temperature, where additive differences are the natural scale.

For comparison, radar dBZ is *already* a log-scale quantity
($\mathrm{dBZ} = 10\log_{10} Z$), so z-scoring or a linear rescale
gives a tractable training signal. **Precip in millimetres is the
linear-scale analogue, and z-scoring is not the right operation.**

### 4.1 Quantifying the imbalance

With Europa stats, the per-channel implicit MSE weight under `v1` is:

| Channel | σ (Europa) | Loss weight $1/\sigma^2$ | Relative (T2 = 1) |
|---|---:|---:|---:|
| T2         | 8.24 K   | 0.01471 | 1×    |
| U10        | 3.43 m/s | 0.08498 | 5.8×  |
| V10        | 3.26 m/s | 0.09411 | 6.4×  |
| precip_1hr | 0.667 mm | 2.2475  | **152.8×** |

A 1 K underestimate of T2 and a 1 mm underestimate of precip contribute
0.0147 and 2.25 to the MSE respectively. The optimizer will therefore
allocate roughly 150× the effective gradient budget to the precip
channel, at the cost of the three Gaussian channels.

### 4.2 Why the imbalance is also a *training-stability* problem

The 152× ratio is the steady-state loss-weight imbalance. The
instantaneous gradient at a single pixel is worse: zero-inflated
heavy-tailed targets in z-scored space produce normalized values up to
$\sim 75$ for a 50 mm/h event (vs. a typical T2 deviation of $\sim 2$).
The squared error of such a pixel is $\sim 75^2 = 5625$, so a single
mis-predicted convective pixel can dominate the gradient of an entire
minibatch. This is the canonical failure mode of training a Gaussian
likelihood on a log-normal target.

---

## 5. Why `v2` doesn't transfer

If you flip the config to `normalization: v2` on Europa, the loader
silently does the following:

* `eastward_wind_10m` / `northward_wind_10m`: branches **match**;
  center/scale forced to (0, 20). But CWA picked σ=20 with Taiwanese
  typhoons in mind (gusts > 30 m/s). Würzburg 10-m winds rarely exceed
  15 m/s. σ=20 therefore *underweights* winds on Europa.
* `maximum_radar_reflectivity`: branch **does not match** — the channel
  is named `precipitation_amount_1hr`. `v2` falls back to the
  empirical stats from the store, i.e. identical to `v1` for this
  channel. The 152× imbalance is unchanged.

`v2` on Europa is therefore strictly worse than `v1` for the regression
stage: it changes nothing about precip, and it actively underweights
winds.

---

## 6. Candidate solutions

Three families of solutions cover the design space. They are not
mutually exclusive but should be evaluated separately.

### A. Linear per-channel rescaling

Keep the dataloader linear, but pick a more physically meaningful
`(μ, σ)` for precip than the empirical mean/std. For example
`μ=0, σ=5` would place a 5 mm/h event at a normalized value of 1 and
move zero to 0, putting precip on the same scale as the other
channels.

* Pros — simplest possible change (one entry in a dict); invertible;
  no Jensen bias; the network still predicts the conditional mean of
  precip in physical units.
* Cons — leaves the heavy tail intact. A 50 mm/h convective cell
  still maps to a normalized value of 10, which contributes ~100× the
  MSE of a 1σ T2 deviation. Training stability is improved but the
  fundamental scale mismatch with the log-normal tail remains.

### B. Variance-stabilizing transform + per-channel rescaling

Apply a non-linear, monotone transform $g(\cdot)$ to the precip channel
that approximately Gaussianises the conditional positive part, then
z-score the transformed field. The standard choices in the
precipitation literature are:

* **log1p**: $g(x) = \log(1+x)$, $g^{-1}(z) = \exp(z) - 1$. Maps
  $0 \mapsto 0$ exactly, so the zero-inflation is preserved at the
  natural origin. Compresses heavy tails strongly: 0.1 mm → 0.095,
  1 mm → 0.69, 10 mm → 2.40, 50 mm → 3.93.
* **asinh-α**: $g(x) = \mathrm{asinh}(\alpha x)$. Smooth through zero,
  asymptotically linear-then-log. Used by MetNet-2/3 and several
  GraphCast variants. With $\alpha = 1$: 1 mm → 0.88, 10 mm → 3.00,
  50 mm → 4.60.
* **Power**: $g(x) = x^{1/3}$. Classical Box-Cox; used by DGMR
  (Ravuri et al.) and PySTEPS. Less aggressive tail compression.

All three are monotonic and invertible. log1p and asinh are
near-identical for the tail (both grow as $\log x$ for large $x$); the
choice between them is largely aesthetic. We recommend **log1p** for
the following reasons:

1. It maps the natural zero of precipitation to zero exactly,
   preserving the qualitative distinction between "no rain" and
   "any rain" without distortion at the origin.
2. The conditional distribution of $\log(\mathrm{precip}\mid \mathrm{precip}>0)$
   is empirically close to Gaussian — log1p is therefore an
   approximate variance-stabilizing transform for the *physical*
   distribution, not just a numerical trick.
3. It corresponds to the standard "multiplicative-error" assumption
   used in operational precipitation verification (e.g. log-RMSE).

After log1p, a sensible per-channel rescaling is `center=0, scale=1`
(i.e. identity post-transform): the resulting normalized values lie
mostly in $[0, 2]$, with rare large-event values up to $\sim 4$,
matching the dynamic range of the other channels.

* Pros — addresses both the loss-weight imbalance *and* the
  heavy-tail training instability; aligns with the published practice
  in the precipitation-ML literature; the network's output range
  matches the other channels.
* Cons — introduces a small Jensen bias (Section 7). The transform is
  non-trivial to back-propagate through evaluation pipelines; both
  `normalize_output` and `denormalize_output` need to know about it.

### C. Per-channel loss reweighting

Leave the dataloader untouched and multiply the per-channel
contribution to $\mathcal{L}_\mathrm{reg}$ by an explicit weight
$\lambda_c$:

$$
\mathcal{L}_\mathrm{reg} = \sum_c \lambda_c \, \mathbb{E}_\mathrm{pixel}\;\big(\hat z_c - z_c\big)^2,
\qquad \lambda_\mathrm{precip} \ll 1.
$$

* Pros — clean separation between data preprocessing and training
  objective; the model still learns the conditional mean of precip in
  physical units; trivially toggleable.
* Cons — does not address the heavy tail (extreme-event pixels still
  produce large gradients within the precip channel); $\lambda_c$ is
  an unprincipled hyperparameter; CorrDiff's `RegressionLoss`
  implementation in `physicsnemo` does not currently expose per-channel
  weights, so this requires either a wrapper loss or a patch to
  upstream code.

---

## 7. Recommendation: **log1p + center=0, scale=1** (option B)

For a thesis-defensible baseline, apply log1p to the precipitation
channel before normalization and use `center=0, scale=1` for the
transformed field. Anchor winds at zero with a Europe-appropriate
scale (`center=0, scale=8`). Keep T2 at its empirical (μ, σ).

Concretely, this means defining a new normalization variant — call it
`v3_europa` — that returns:

| Channel | center | scale | Transform |
|---|---:|---:|---|
| `temperature_2m`           | 283.24 | 8.24 | identity |
| `eastward_wind_10m`        | 0      | 8    | identity |
| `northward_wind_10m`       | 0      | 8    | identity |
| `precipitation_amount_1hr` | 0      | 1    | **log1p** |

The transform changes how `normalize_output` and `denormalize_output`
are implemented (a per-channel pre- and post-transform must be applied
around the existing z-score). The numbers above are starting points,
not optimal values; they can be refined empirically by inspecting the
per-channel std of the normalized training data and the convergence
behaviour of the regression loss.

### 7.1 Justification

* **Equal scale across channels.** After the transform, each channel's
  normalized values lie roughly in $[-3, +3]$, matching the dynamic
  range under which the UNet was designed and originally trained.
* **No silent re-weighting.** The cross-channel loss balance is
  determined by an explicit, documentable choice rather than by the
  empirical std of a zero-inflated distribution.
* **No tail blow-up.** A 50 mm/h convective pixel maps to
  $\log(1+50) \approx 3.93$, comparable to a 3σ T2 deviation. The
  optimizer is no longer driven by single-pixel storms.
* **Precedent.** Variance-stabilizing transforms for precipitation are
  the standard in the precipitation-ML literature (MetNet-2/3, DGMR,
  NowcastNet, NeuralLAM-precip, and most operational nowcasting
  systems). The CWA `v2` overrides for radar dBZ are conceptually the
  same operation (taming a non-Gaussian channel) applied to a
  log-scale variable, which is why a linear rescale was sufficient
  there.

### 7.2 Jensen-bias caveat

Because $\log(1+x)$ is concave, by Jensen's inequality:

$$
\exp\!\Big(\mathbb{E}[\log(1+Y)]\Big) - 1 \;\leq\; \mathbb{E}[Y].
$$

If the regression network learns the conditional mean of
$\log(1+Y)$ given the low-res input, the back-transformed prediction
$\exp(\hat z) - 1$ systematically *under-predicts* the conditional mean
of $Y$. The magnitude of the bias scales with the conditional
variance of $\log(1+Y)$.

Two things to know:

1. **In the full two-stage CorrDiff model**, the regression is *not*
   required to be an unbiased estimator of $\mathbb{E}[Y\mid x_\mathrm{lr}]$.
   The residual diffusion stage models the full conditional
   distribution of $Y$ around the regression output, and its samples
   correctly recover the conditional mean (in physical units) by
   construction. The Jensen bias is therefore absorbed by the
   downstream stage and does not propagate to the final outputs.
2. **For a regression-only baseline**, the bias is real and should be
   reported. It can be measured by computing the empirical mean of
   $Y - (\exp(\hat z)-1)$ on the validation split and, if desired,
   corrected by a one-parameter multiplicative offset
   $\hat Y_\mathrm{corr} = c \cdot (\exp(\hat z) - 1)$ with $c$ chosen
   to zero the mean residual on validation. A more principled
   correction is the lognormal smearing estimator (Duan 1983), which
   uses the empirical distribution of residuals rather than assuming
   lognormality.

For this thesis, we recommend training the regression with log1p,
**reporting the Jensen bias on the 256-timestamp validation set**,
and noting that the bias is automatically corrected once the residual
diffusion stage is added.

---

## 8. Implementation plan

A minimal patch to add `v3_europa`:

1. **In [`datasets/cwb.py`](datasets/cwb.py)**, add:
   ```python
   def get_target_normalizations_v3_europa(group):
       """Europa-tuned normalization with log1p on precipitation."""
       variable = group["cwb_variable"][:]
       center = np.array(group["cwb_center"][:], dtype=np.float32)
       scale  = np.array(group["cwb_scale"][:],  dtype=np.float32)
       center = np.where(variable == "eastward_wind_10m",        0.0, center)
       center = np.where(variable == "northward_wind_10m",       0.0, center)
       center = np.where(variable == "precipitation_amount_1hr", 0.0, center)
       scale  = np.where(variable == "eastward_wind_10m",        8.0, scale)
       scale  = np.where(variable == "northward_wind_10m",       8.0, scale)
       scale  = np.where(variable == "precipitation_amount_1hr", 1.0, scale)
       return center, scale
   ```
2. **Register the variant** in `get_zarr_dataset`'s normalization dict
   (currently lines 544-548) so `normalization: v3_europa` is selectable
   from the YAML config.
3. **Add the log1p / expm1 transform** for the precipitation channel.
   This requires changes to `_ZarrDataset.normalize_output` and
   `denormalize_output` (and the equivalent wrappers in `ZarrDataset`).
   The cleanest design is a per-channel pre/post-transform table read
   from the normalization function, e.g. by having
   `get_target_normalizations_v3_europa` *also* return a list of
   per-channel transform functions:
   ```python
   return center, scale, [None, None, None, np.log1p], [None, None, None, np.expm1]
   ```
   and updating the existing call sites to apply them. The signature
   change touches `cwb.py`, but no model or training-loop code.
4. **Recompute or accept the Jensen bias** as discussed in 7.2 — the
   regression's output should be reported both raw (in transformed
   units) and back-transformed (with bias correction documented).
5. **In the config** ([`conf/config_training_europa_regression-alex.yaml`](conf/config_training_europa_regression-alex.yaml)),
   set `dataset.normalization: v3_europa` once the loader change is
   in place.

---

## 9. Empirical validation plan

A defensible thesis result will include:

* **Baseline (`v1`)** — run regression as-is, document the loss
  imbalance, log per-channel validation MSE in *physical* units (not
  normalized) so the imbalance is interpretable.
* **`v3_europa`** — run the same architecture with the recommended
  log1p + center=0 scheme, compare per-channel validation MSE in
  physical units, and report whether the loss is now balanced.
* **Per-event analysis** — for the curated 256 validation timestamps
  (`conf/val_times_2021.yaml`), plot the per-pixel error distribution
  for precip under both schemes; the log1p variant should show a
  much-tighter distribution and a documentable Jensen offset on the
  mean.
* **Diffusion stage** — once the residual diffusion model is trained
  on top of the regression, the Jensen bias should be absorbed; this
  is the headline result that justifies the choice.

These four comparisons (loss curves, per-channel physical MSE, error
distributions, downstream diffusion behaviour) collectively answer
the thesis question of whether `v3_europa` is the right design choice
for European precipitation downscaling.

---

## 10. References (informal)

* Mardani et al., *Generative Residual Diffusion Modeling for Km-Scale
  Atmospheric Downscaling*, 2024 — the CorrDiff paper; describes the
  two-stage regression + residual diffusion design.
* Sønderby et al., *MetNet*, 2020; Andrychowicz et al., *MetNet-2*,
  2023; *MetNet-3*, 2023 — use asinh and similar transforms for
  precipitation channels.
* Ravuri et al., *Skilful precipitation nowcasting using deep generative
  models of radar* (DGMR), Nature 2021 — cube-root preprocessing.
* Zhang et al., *NowcastNet*, Nature 2023 — physics-informed
  precipitation modelling, uses log-domain reasoning.
* Duan, *Smearing estimate: a nonparametric retransformation method*,
  JASA 1983 — the classical retransformation-bias correction.
* Anscombe, *The transformation of Poisson, binomial and
  negative-binomial data*, Biometrika 1948 — origin of
  variance-stabilizing transforms.
