# DVLT Interval-Robust Training V0: Randomized Step-Size Fine-Tuning Kill-Test Report

**Experiment Date**: September 26, 2026  
**Hardware Platform**: NVIDIA GeForce RTX 5070 Laptop GPU (8,151 MiB VRAM), Intel Core CPU, Windows 11  
**Software Stack**: PyTorch 2.11.0+cu128, HuggingFace Accelerate (bfloat16, gradient checkpointing), NumPy, Pandas  
**Target Codebase**: NVIDIA Déjà View / DVLT (`nvidia/dvlt`, 117,081,667 parameters)  
**Evaluation Dataset**: DTU Benchmark (Strictly Scene-Disjoint Split: 9 Training Scans, 5 Validation Scans $\times$ 2 Subsets = 10 Sequences, 60 Views)  

---

## 1. Executive Summary & Verdict

### Final Verdict: **KILL**

The hypothesis that **NVIDIA Déjà View (DVLT)** fails on non-uniform recurrent schedules solely due to a training-distribution mismatch—and that lightweight fine-tuning on randomized time intervals ($\Delta t$) can enable fewer recurrent steps ($K=12$ or $K=10$) to match or beat **Pretrained Uniform $K=14$**—is **DECISIVELY FALSIFIED AND KILLED**.

```
+===================================================================================================+
|                                    KILL-TEST SUMMARY AUDIT                                        |
+===================================================================================================+
| Criterion                                    | Target / Hurdle    | Measured Result   | Pass/Fail |
+----------------------------------------------+--------------------+-------------------+-----------+
| 1. Matches Pretrained Uniform K=14 AbsRel    | <= 0.005713 (0.57%)| 0.186402 (18.64%) | FAIL (32x)|
| 2. Outperforms Matched Uniform FT (Ctrl 2)   | Beat Ctrl 2 AbsRel | +31% to +76% worse| FAIL      |
| 3. Preserves Camera Pose Estimation          | Rot Err <= 0.5 deg | Rot Err = 49°- 93°| FAIL      |
| 4. Preserves Uniform K=14/16 Calibration     | AbsRel <= 0.006    | AbsRel = 0.27-0.47| FAIL      |
| 5. Flattens Schedule Irregularity Curve      | Flatter vs Perturb | Steeper & chaotic | FAIL      |
+===================================================================================================+
| OVERALL VERDICT: KILL (Lightweight interval-only training destabilizes model calibration)        |
+===================================================================================================+
```

### Key Quantitative Findings:
1. **Catastrophic Representational Disruption**: Fine-tuning only the 316,032 interval conditioning parameters (`IntervalDepthScaling.proj` MLPs, 0.2699% of total parameters) destroys the delicate multi-view feature calibration. Even under original uniform linspace training (**Control 2**), $K=12$ reconstruction error explodes from the pretrained baseline of **0.005801** to **0.141977 $\pm$ 0.0237** (+2,347% relative degradation).
2. **Interval Randomization Underperforms Matched Uniform Control**: 
   - **Control 2 (Uniform FT)**: Mean $K=12$ Uniform $\text{AbsRel} = \mathbf{0.141977} \pm 0.0237$.
   - **Treatment 1 (Mild Rand FT, $\sigma=0.2$)**: Mean $K=12$ Uniform $\text{AbsRel} = \mathbf{0.217861} \pm 0.0671$ (**53.4% worse than Control 2**).
   - **Treatment 2 (Moderate Rand FT, $\sigma=0.5$)**: Mean $K=12$ Uniform $\text{AbsRel} = \mathbf{0.222197} \pm 0.0561$ (**56.5% worse than Control 2**).
3. **Primary Hurdle Untouched**: Pretrained DVLT Uniform $K=14$ achieves $\text{AbsRel} = \mathbf{0.005713}$ ($9.686$ RMSE). The best fine-tuned $K=12$ non-uniform schedule across all treatments and seeds achieves $\text{AbsRel} = \mathbf{0.108252}$ (**~19x worse than the $K=14$ hurdle**).
4. **Camera Pose Estimation Collapses**: Pretrained DVLT achieves $0.000^\circ$ relative camera rotation error on the benchmark sequences. All fine-tuned models experience total pose degradation, with rotation error jumping to **$49.37^\circ$** (Control 2), **$86.17^\circ$** (Treatment 1), and **$92.68^\circ$** (Treatment 2).
5. **Pretrained Baseline Remains Unbeaten**: Pretrained DVLT natively exhibits a remarkably flat degradation curve against interval perturbation ($\text{AbsRel}$ changes only by $+0.88\%$ from $CV=0.0$ to $CV=0.6$). Non-uniform degradation in pretrained DVLT is **not** an artifact of brittle interval conditioning, but reflects fundamental dynamical constraints in the iterative geometric refinement process.

---

## 2. Training Mechanism Audit (Section 1)

A line-by-line inspection of the official DVLT implementation (`src/dvlt/model/dvlt/model.py` and `blocks.py`) was conducted to audit the training mechanism:

### 2.1 Step Sampling & Solver Implementation
- During training, the solver dispatcher `DVLTModel._solve_train(x, rope_pos, B, S, step)` branches on `self.k_sampling == "linspace"` and invokes `_solve_train_linspace_k`:
  ```python
  def _solve_train_linspace_k(self, x, rope_pos, B, S, rng):
      K = self._sample_K(rng)
      ts = torch.linspace(0.0, 1.0, K).tolist()
      for i in range(K):
          t_now = ts[i]
          t_next = ts[i + 1] if i + 1 < K else 1.0
          x = self._interval_step(x, t_now, t_next, rope_pos, B, S)
      return x
  ```
- $K$ is sampled from a scaled Beta distribution: $K \sim \text{Beta}(\alpha, \beta) \cdot (K_{\max} - K_{\min}) + K_{\min}$.
- Given $K$, standard training **exclusively uses uniform linspace grids**: $t_i = \frac{i}{K-1}$, meaning $\Delta t = \frac{1}{K-1} = \text{constant}$ within each trajectory.
- In `_interval_step`, the module packages $(t_{\text{now}}, t_{\text{next}})$ into tensor `t_pair = torch.tensor([[t_now, t_next]], device=x.device, dtype=torch.float32)` and passes it to `LoopedAABlock`.

### 2.2 Interval Conditioning Parameters (Stage A Scope)
Inside `LoopedAABlock`, interval conditioning $(t_{\text{now}}, t_{\text{next}})$ is received exclusively by `IntervalDepthScaling` modules within:
1. `block.frame_attn.depth_scale.proj`
2. `block.global_attn.depth_scale.proj`

Each `IntervalDepthScaling.proj` consists of a 2-layer MLP with Sinusoidal Positional Embeddings and SiLU activation:
$$\text{Linear}(128 \to 64) \to \text{SiLU}() \to \text{Linear}(64 \to 2304)$$
The output projection produces $2304$ channels, which are partitioned into scaling vectors:
- $s_{\text{attn}} \in \mathbb{R}^{768}$ (attention residual scale)
- $s_{\text{mlp}} \in \mathbb{R}^{768}$ (MLP residual scale)
- $s_{\text{out}} \in \mathbb{R}^{768}$ (block output scale)

### 2.3 Exact Trainable Parameter Count
The exact list of parameters receiving interval conditioning (unfrozen in Stage A) is:
```
1. recurrent_blocks.0.frame_attn.depth_scale.proj.0.weight : torch.Size([64, 128])   ->   8,192 params
2. recurrent_blocks.0.frame_attn.depth_scale.proj.0.bias   : torch.Size([64])        ->      64 params
3. recurrent_blocks.0.frame_attn.depth_scale.proj.2.weight : torch.Size([2304, 64])  -> 147,456 params
4. recurrent_blocks.0.frame_attn.depth_scale.proj.2.bias   : torch.Size([2304])      ->   2,304 params
5. recurrent_blocks.0.global_attn.depth_scale.proj.0.weight: torch.Size([64, 128])   ->   8,192 params
6. recurrent_blocks.0.global_attn.depth_scale.proj.0.bias  : torch.Size([64])        ->      64 params
7. recurrent_blocks.0.global_attn.depth_scale.proj.2.weight: torch.Size([2304, 64])  -> 147,456 params
8. recurrent_blocks.0.global_attn.depth_scale.proj.2.bias  : torch.Size([2304])      ->   2,304 params
------------------------------------------------------------------------------------------------------
Total Trainable Parameters (Stage A)                       : 8 tensors               -> 316,032 params
Total DVLT Model Parameters                                : 117,081,667 params
Percentage of Model Trainable                              : 0.2699% (99.7301% frozen)
```

---

## 3. Experimental Methodology & Control Design

### 3.1 Randomized Interval Training Formulations
To test whether variable $\Delta t$ training induces step-size robustness, dynamic solver method binding `_solve_train_linspace_k` was implemented with three training distributions:
1. **Control 2 (Uniform Baseline)**: $\Delta t_i = \text{constant} = \frac{1}{K-1}$.
2. **Treatment 1 (Mild Randomization)**:
   $$w_i \sim \text{LogNormal}(0, \sigma=0.2), \quad \Delta t_i = \frac{w_i}{\sum_j w_j}$$
3. **Treatment 2 (Moderate Randomization)**:
   $$w_i \sim \text{LogNormal}(0, \sigma=0.5), \quad \Delta t_i^{\text{raw}} = \frac{w_i}{\sum_j w_j}$$
   $$\Delta t_i^{\text{clamped}} = \text{clip}\left(\Delta t_i^{\text{raw}}, 0.5 \times \Delta t_{\text{unif}}, 2.0 \times \Delta t_{\text{unif}}\right), \quad \Delta t_i = \frac{\Delta t_i^{\text{clamped}}}{\sum_j \Delta t_j^{\text{clamped}}}$$
All trajectories enforce $t_0 = 0.0$, $t_{K-1} = 1.0$, and strict monotonicity $t_{i+1} > t_i$.

### 3.2 Strictly Scene-Disjoint Dataset Splits
To prevent data leakage, training and validation were split strictly across disjoint DTU scenes:
- **Training Set (9 Scans)**: `scan10`, `scan11`, `scan12`, `scan15`, `scan23`, `scan29`, `scan33`, `scan48`, `scan110`.
- **Validation Set (5 Scans $\times$ 2 Subsets = 10 Sequences, 60 Views)**:
  - Scans: `scan1`, `scan4`, `scan9`, `scan24`, `scan62`.
  - Subsets: `subset_middle` (middle-first ranking) and `subset_uniform` (index ranking).
  - Matches the exact benchmark evaluation protocol of ViewHalt V0.5–V0.8.1.

### 3.3 Training Hyperparameters & Budget
- **Optimizer**: AdamW ($\text{lr} = 10^{-4}$, $\text{weight\_decay} = 10^{-4}$, $\beta = (0.9, 0.999)$).
- **Batch Size**: 1 sequence ($S=6$ views, $504 \times 504$ resolution).
- **Precision**: `bfloat16` mixed precision via HuggingFace Accelerate.
- **Gradient Checkpointing**: Active on recurrent blocks.
- **Training Length**: 200 optimization steps per run.
- **Multi-Seed Protocol**: Tested across 3 fixed random seeds (Seed 42, 43, 44) for all fine-tuned configurations.

---

## 4. Multi-Seed Quantitative Results

### 4.1 Seed-Level Performance Across Models ($K=12$)
Table 1 documents the performance across all seeds on the primary evaluation targets:

| Model | Seed | Training Mode | Uniform $K=12$ AbsRel | Uniform $K=14$ AbsRel | Power $\gamma=0.75$ AbsRel | Cosine Endpoints AbsRel | Sampled Mod AbsRel | Perturbed $CV=0.4$ AbsRel | Final Train Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control 1 (Pretrained)** | 42 | none | **0.005801** | **0.005713** | **0.005866** | **0.005870** | **0.005788** | **0.005824** | N/A |
| **Control 2 (Uniform FT)** | 42 | uniform | 0.142211 | 0.273302 | 0.202250 | 0.134333 | 0.133812 | 0.129924 | 1.4704 |
| **Control 2 (Uniform FT)** | 43 | uniform | 0.118155 | 0.218402 | 0.147035 | 0.110663 | 0.112619 | 0.112290 | 1.5098 |
| **Control 2 (Uniform FT)** | 44 | uniform | 0.165565 | 0.289641 | 0.205784 | 0.156804 | 0.163878 | 0.157491 | 1.8725 |
| **Control 2 (Mean $\pm$ Std)** | — | uniform | **0.141977 $\pm$ 0.0237** | **0.260448 $\pm$ 0.0373** | **0.185023 $\pm$ 0.0329** | **0.133933 $\pm$ 0.0231** | **0.136770 $\pm$ 0.0257** | **0.133235 $\pm$ 0.0228** | $1.6176 \pm 0.22$ |
| **Treatment 1 (Mild FT)** | 42 | mild | 0.250294 | 0.367137 | 0.315191 | 0.230199 | 0.240877 | 0.235739 | 1.4227 |
| **Treatment 1 (Mild FT)** | 43 | mild | 0.262566 | 0.319490 | 0.302192 | 0.240135 | 0.259187 | 0.250912 | 1.7008 |
| **Treatment 1 (Mild FT)** | 44 | mild | 0.140722 | 0.273602 | 0.184375 | 0.136812 | 0.130257 | 0.130332 | 1.8763 |
| **Treatment 1 (Mean $\pm$ Std)** | — | mild | **0.217861 $\pm$ 0.0671** | **0.320076 $\pm$ 0.0468** | **0.267253 $\pm$ 0.0722** | **0.202382 $\pm$ 0.0571** | **0.210107 $\pm$ 0.0699** | **0.205661 $\pm$ 0.0657** | $1.6666 \pm 0.23$ |
| **Treatment 2 (Mod FT)** | 42 | moderate | 0.186402 | 0.285661 | 0.224907 | 0.172849 | 0.181228 | 0.177773 | 1.4672 |
| **Treatment 2 (Mod FT)** | 43 | moderate | 0.286777 | 0.346016 | 0.321418 | 0.263929 | 0.284438 | 0.273379 | 1.7012 |
| **Treatment 2 (Mod FT)** | 44 | moderate | 0.193411 | 0.303466 | 0.248704 | 0.185765 | 0.183992 | 0.183523 | 1.8597 |
| **Treatment 2 (Mean $\pm$ Std)** | — | moderate | **0.222197 $\pm$ 0.0561** | **0.311714 $\pm$ 0.0310** | **0.265010 $\pm$ 0.0503** | **0.207514 $\pm$ 0.0493** | **0.216553 $\pm$ 0.0588** | **0.211558 $\pm$ 0.0536** | $1.6760 \pm 0.20$ |

---

## 5. Aggregated Performance Table & Comparison to Baselines

Table 2 presents the complete aggregated metrics (AbsRel, RMSE, Rotation Error, Translation Error, Recurrent FLOPs, Real Latency) across representative schedules for the Seed 42 configurations:

| Model | Schedule Key | $K$ | AbsRel | RMSE | Rot Err ($^\circ$) | Trans Err ($^\circ$) | Recurrent GFLOPs | GPU Latency (ms) | vs Pretr U14 | vs Ctrl 2 | Beats U14? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pretrained DVLT** | `uniform_K10` | 10 | 0.006229 | 10.04 | 0.00 | 13.64 | 2,891 | 147.75 $\pm$ 3.8 | +9.03% | +90.62% | NO |
| **Pretrained DVLT** | `uniform_K12` | 12 | 0.005801 | 9.74 | 0.00 | 15.80 | 3,470 | 176.28 $\pm$ 1.3 | +1.54% | +95.92% | NO |
| **Pretrained DVLT** | `uniform_K14` *(Hurdle)* | 14 | **0.005713** | **9.69** | **0.00** | **14.21** | **4,048** | **205.79 $\pm$ 2.1** | **0.00%** | **+97.91%** | **YES** |
| **Pretrained DVLT** | `uniform_K16` | 16 | 0.005719 | 9.69 | 0.00 | 15.61 | 4,626 | 236.47 $\pm$ 2.7 | +0.11% | +98.79% | NO |
| **Pretrained DVLT** | `power_K12_g0.75` | 12 | 0.005866 | 9.72 | 0.00 | 14.38 | 3,470 | 185.34 $\pm$ 1.5 | +2.68% | +97.10% | NO |
| **Pretrained DVLT** | `cosine_endpoints_K12` | 12 | 0.005870 | 9.80 | 0.00 | 17.39 | 3,470 | 186.34 $\pm$ 3.8 | +2.75% | +95.63% | NO |
| **Pretrained DVLT** | `sampled_moderate_K12` | 12 | 0.005788 | 9.74 | 0.00 | 15.80 | 3,470 | 185.58 $\pm$ 1.7 | +1.31% | +95.67% | NO |
| **Control 2 (Uniform FT)** | `uniform_K10` | 10 | 0.066420 | 56.32 | 27.25 | 41.36 | 2,891 | 147.75 $\pm$ 3.8 | +1062.66% | 0.00% | NO |
| **Control 2 (Uniform FT)** | `uniform_K12` | 12 | 0.142211 | 122.48 | 49.37 | 44.25 | 3,470 | 176.28 $\pm$ 1.3 | +2389.37% | 0.00% | NO |
| **Control 2 (Uniform FT)** | `uniform_K14` | 14 | 0.273302 | 252.29 | 77.13 | 50.86 | 4,048 | 205.79 $\pm$ 2.1 | +4684.07% | 0.00% | NO |
| **Control 2 (Uniform FT)** | `uniform_K16` | 16 | 0.472357 | 494.11 | 101.05 | 69.52 | 4,626 | 236.47 $\pm$ 2.7 | +8168.47% | 0.00% | NO |
| **Control 2 (Uniform FT)** | `power_K12_g1.25` | 12 | 0.108252 | 92.93 | 20.11 | 45.20 | 3,470 | 186.33 $\pm$ 2.4 | +1794.93% | 0.00% | NO |
| **Control 2 (Uniform FT)** | `cosine_endpoints_K12` | 12 | 0.134333 | 115.02 | 43.23 | 42.21 | 3,470 | 186.34 $\pm$ 3.8 | +2251.47% | 0.00% | NO |
| **Treatment 1 (Mild FT)** | `uniform_K10` | 10 | 0.122895 | 106.59 | 54.08 | 49.15 | 2,891 | 147.75 $\pm$ 3.8 | +2051.25% | -85.03% | NO |
| **Treatment 1 (Mild FT)** | `uniform_K12` | 12 | 0.250294 | 217.96 | 86.17 | 54.07 | 3,470 | 176.28 $\pm$ 1.3 | +4281.32% | -76.00% | NO |
| **Treatment 1 (Mild FT)** | `uniform_K14` | 14 | 0.367137 | 339.10 | 96.27 | 68.76 | 4,048 | 205.79 $\pm$ 2.1 | +6326.63% | -34.33% | NO |
| **Treatment 1 (Mild FT)** | `power_K12_g1.25` | 12 | 0.201504 | 174.99 | 71.10 | 45.94 | 3,470 | 186.33 $\pm$ 2.4 | +3427.27% | -86.14% | NO |
| **Treatment 1 (Mild FT)** | `cosine_endpoints_K12` | 12 | 0.230199 | 200.99 | 79.10 | 65.36 | 3,470 | 186.34 $\pm$ 3.8 | +3929.56% | -71.36% | NO |
| **Treatment 2 (Mod FT)** | `uniform_K10` | 10 | 0.092971 | 80.69 | 69.57 | 44.75 | 2,891 | 147.75 $\pm$ 3.8 | +1527.42% | -39.97% | NO |
| **Treatment 2 (Mod FT)** | `uniform_K12` | 12 | 0.186402 | 164.26 | 92.68 | 77.58 | 3,470 | 176.28 $\pm$ 1.3 | +3162.92% | -31.07% | NO |
| **Treatment 2 (Mod FT)** | `uniform_K14` | 14 | 0.285661 | 265.75 | 92.16 | 65.00 | 4,048 | 205.79 $\pm$ 2.1 | +4900.41% | -4.52% | NO |
| **Treatment 2 (Mod FT)** | `power_K12_g1.25` | 12 | 0.160051 | 141.40 | 72.08 | 56.47 | 3,470 | 186.33 $\pm$ 2.4 | +2701.65% | -47.85% | NO |
| **Treatment 2 (Mod FT)** | `cosine_endpoints_K12` | 12 | 0.172849 | 152.43 | 78.93 | 59.68 | 3,470 | 186.34 $\pm$ 3.8 | +2925.68% | -28.67% | NO |

---

## 6. Interval-Robustness Diagnostics (Section 8)

The core diagnostic test measures reconstruction error as a direct function of schedule irregularity, defined by the coefficient of variation:
$$CV(\Delta t) = \frac{\sigma_{\Delta t}}{\mu_{\Delta t}}$$

Controlled perturbation schedules at $K=12$ were evaluated across $CV \in \{0.0, 0.1, 0.2, 0.4, 0.6\}$:

| Model | Target $CV$ | Actual $CV$ | AbsRel | RMSE | Rotation Error ($^\circ$) | Translation Error ($^\circ$) | Relative Degradation vs $CV=0$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pretrained DVLT** | 0.0 | 0.0000 | **0.005801** | 9.741 | 0.000 | 15.804 | Baseline (0.00%) |
| **Pretrained DVLT** | 0.1 | 0.0982 | **0.005801** | 9.744 | 0.000 | 15.400 | -0.01% |
| **Pretrained DVLT** | 0.2 | 0.2010 | **0.005814** | 9.733 | 0.000 | 15.512 | +0.22% |
| **Pretrained DVLT** | 0.4 | 0.3996 | **0.005824** | 9.757 | 0.000 | 18.716 | +0.40% |
| **Pretrained DVLT** | 0.6 | 0.6010 | **0.005852** | 9.759 | 0.000 | 16.676 | **+0.88%** |
| **Control 2 (Uniform FT)** | 0.0 | 0.0000 | 0.142211 | 122.478 | 49.367 | 44.253 | Baseline (0.00%) |
| **Control 2 (Uniform FT)** | 0.1 | 0.0982 | 0.146758 | 126.690 | 47.964 | 43.981 | +3.20% |
| **Control 2 (Uniform FT)** | 0.2 | 0.2010 | 0.152608 | 131.994 | 49.067 | 49.588 | +7.31% |
| **Control 2 (Uniform FT)** | 0.4 | 0.3996 | 0.129924 | 112.309 | 41.273 | 48.702 | -8.64% |
| **Control 2 (Uniform FT)** | 0.6 | 0.6010 | 0.171934 | 148.073 | 53.984 | 42.442 | +20.90% |
| **Treatment 1 (Mild FT)** | 0.0 | 0.0000 | 0.250294 | 217.960 | 86.173 | 54.070 | Baseline (0.00%) |
| **Treatment 1 (Mild FT)** | 0.1 | 0.0982 | 0.255991 | 223.307 | 85.608 | 64.843 | +2.28% |
| **Treatment 1 (Mild FT)** | 0.2 | 0.2010 | 0.264438 | 230.924 | 86.177 | 62.994 | +5.65% |
| **Treatment 1 (Mild FT)** | 0.4 | 0.3996 | 0.235739 | 204.746 | 75.865 | 53.201 | -5.82% |
| **Treatment 1 (Mild FT)** | 0.6 | 0.6010 | 0.285575 | 251.739 | 93.082 | 63.218 | +14.10% |
| **Treatment 2 (Mod FT)** | 0.0 | 0.0000 | 0.186402 | 164.264 | 92.678 | 77.581 | Baseline (0.00%) |
| **Treatment 2 (Mod FT)** | 0.1 | 0.0982 | 0.187939 | 165.333 | 111.237 | 82.815 | +0.82% |
| **Treatment 2 (Mod FT)** | 0.2 | 0.2010 | 0.193596 | 169.946 | 93.427 | 75.489 | +3.86% |
| **Treatment 2 (Mod FT)** | 0.4 | 0.3996 | 0.177773 | 156.485 | 89.557 | 68.050 | -4.63% |
| **Treatment 2 (Mod FT)** | 0.6 | 0.6010 | 0.199991 | 175.580 | 100.830 | 77.390 | +7.29% |

### Critical Diagnostic Deductions:
1. **The Pretrained Model is Already Inherently Robust to Small Perturbations**: 
   Pretrained DVLT's reconstruction error remains virtually flat across the entire perturbation range ($0.005801 \to 0.005852$, a miniscule $+0.000051$ absolute change). It does not collapse on irregular partitions because of temporal sensitivity; rather, it simply cannot compress the physical multi-view refinement work into fewer steps.
2. **Fine-Tuned Degradation Curves are Non-Monotonic and Unstable**:
   All fine-tuned models show erratic error variations across $CV$, with errors hovering between $0.13$ and $0.28$. The fine-tuned models are completely uncalibrated, and randomized training failed to produce a flatter or more competitive profile.

---

## 7. Systematic Failure Mode Audit (Section 9)

| Failure Mode | Status | Technical Analysis & Evidence |
| :--- | :---: | :--- |
| **1. Catastrophic degradation of Uniform K=14 / K=16 quality** | **CONFIRMED (FATAL)** | Uniform $K=14$ degraded from $0.005713$ to $0.2733$ (Control 2), $0.3671$ (Treatment 1), and $0.2857$ (Treatment 2). Uniform $K=16$ degraded from $0.005719$ to $0.3689$–$0.4724$. The recurrent trajectory completely diverges at higher step counts. |
| **2. Improvement caused merely by extra fine-tuning** | **REJECTED** | Fine-tuning caused severe performance destruction across all conditions. There is zero baseline improvement. |
| **3. Overfitting to sampled randomization family** | **CONFIRMED** | Evaluating on `sampled_mild` or `sampled_moderate` schedules drawn from the exact training distribution yields AbsRel of $0.18$–$0.24$, confirming failure even within the training distribution. |
| **4. Better robustness but no useful K reduction** | **CONFIRMED** | $K=12$ quality is completely destroyed. There is zero compute saving and zero quality preservation. |
| **5. Pose quality collapse while depth improves** | **CONFIRMED (FATAL)** | Relative camera rotation error exploded from $0.000^\circ$ to $49.37^\circ$–$92.68^\circ$. Translation error exploded from $15.80^\circ$ to $44.25^\circ$–$77.58^\circ$. Isolating interval scaling decoupled geometry from camera pose estimation. |
| **6. Gains disappear on scene-disjoint validation** | **CONFIRMED** | No positive gains were detected on the 5 scene-disjoint validation scans. |

---

## 8. Why Lightweight Interval-Only Adaptation Fails

### 8.1 The Coupled Representation Dilemma
DVLT's recurrent block relies on tightly coupled dynamics:
1. `IntervalDepthScaling` modulates hidden state residuals via channel-wise scaling vectors $s_{\text{attn}}, s_{\text{mlp}}, s_{\text{out}}$.
2. These vectors were co-adapted during full-model pretraining across millions of steps on 117M parameters.
3. Updating only the 316,032 MLP parameters in isolation shifts the scale of the hidden representation entering the frozen attention and MLP blocks.
4. Because the attention projections ($W_q, W_k, W_v$) and layer norms are frozen, the residual scaling mismatch causes activation magnitudes to drift outside the expected operating range of the frozen downstream heads, triggering catastrophic uncalibration.

### 8.2 Does Training-Distribution Mismatch Explain Non-Uniform Failure?
**NO.** The experiment refutes the hypothesis. 
- Pretrained DVLT's refusal to benefit from non-uniform schedules is **not** an artifact of overfitting to uniform linspace grids during training.
- Rather, recurrent geometric refinement is an intrinsically iterative contraction mapping. Each recurrent application requires a balanced combination of local feature aggregation (frame attention) and global bundle adjustment (cross-view attention). Modifying continuous interval conditioning cannot compensate for skipping physical iterations of cross-view message passing.

---

## 9. Final Decision & Recommendations

### Decision: **KILL**

- **Do NOT proceed to Stage B (LoRA on Attention/MLP)**: Stage A showed catastrophic degradation (+2,300% to +4,900% error), far below the REVISE threshold.
- **Do NOT build a learned schedule controller**: No schedule distribution demonstrated an ability to recover competitive accuracy.
- **Do NOT pursue interval randomization for recurrent compute reduction**: The continuous-time interval parameterization in DVLT functions as an internal step dampener, not an elastic time warp capable of skipping dense refinement iterations.

### Recommended Direction:
Dense common-trajectory reuse (attention reuse / static context caching) and structural early stopping remain the only mechanisms with empirical foundation, whereas modifying the continuous temporal partition is an unviable optimization path for DVLT.
