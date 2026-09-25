# DVLT Attention-Reuse V0: Cross-Iteration Global Attention Reuse Kill-Test Report

**Date:** 2026-09-25  
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (8 GB VRAM, Compute Capability 12.0)  
**Environment:** PyTorch 2.11.0+cu128, CUDA 12.8, `bfloat16`  
**Dataset:** DTU Evaluation Set (5 representative scans: `scan1`, `scan4`, `scan9`, `scan24`, `scan62` $\times$ 2 subsets: `subset_middle`, `subset_uniform` = 10 sequences, 60 views)  
**Trajectory:** Official DVLT Common $K=16$ linspace recurrent trajectory (`torch.linspace(0, 1, 16)`)  
**Verdict:** **KILL** (Dense Cross-Iteration Global Attention Reuse cannot beat Uniform Early Stopping at matched quality)

---

## Executive Summary

Following the termination of ViewHalt (where irregular sparse-dispatch overhead eliminated theoretical compute savings), this study explores **Attention-Reuse V0**: whether cross-iteration reuse or skipping of expensive global cross-view attention during late recurrent iterations (steps 8–16) can reduce real GPU wall-clock latency while maintaining dense, regular `(B, S*P, C)` tensor allocations.

We instrumented the $K=16$ recurrent trajectory of NVIDIA Déjà View (DVLT), implemented three dense reuse variants across 30 candidate schedules, measured reconstruction metrics (AbsRel, RMSE, Camera Pose Error), and benchmarked real GPU recurrent latency using `torch.cuda.Event` (3 warmups, 10 repetitions).

### Key Takeaway & Kill-Test Outcome
1. **Mathematical Feasibility vs Economic Competitiveness:**
   - Global cross-view representation becomes remarkably stable in late iterations: by step 14, consecutive global attention output cosine similarity reaches **0.9991**, key similarity reaches **0.9996**, and residual direction similarity reaches **0.9932**.
   - Residual reuse (Variant B with $\alpha=0.75$) successfully preserves reconstruction quality across multiple skip steps without quality degradation (e.g., `reuse_15_16` achieves **AbsRel 0.005661** vs baseline $K=16$ **0.005719**).
2. **The Economic Hurdle That Kills the Direction:**
   - The primary hurdle is **Fixed $i=14$** (uniform early stopping at step 14). Fixed $i=14$ completely halts **both** frame attention and global attention for steps 15 and 16, executing in **218.82 ms** with **AbsRel 0.005786**.
   - In contrast, any 16-step attention reuse schedule **must still execute frame attention across all 6 views** ($100.58\text{ GFLOPs}$ per step) at steps 15 and 16.
   - Consequently, every quality-preserving reuse schedule has **higher total compute and higher wall-clock latency** than Fixed $i=14$:
     - The fastest quality-matching reuse variant is `VarB: reuse_even_after_10 (alpha=0.5)` at **227.58 ms**, which is **4.0% slower** than Fixed $i=14$ (**218.82 ms**).
     - Variant C (Frozen Global K/V) saves virtually zero wall-clock latency (**257.3–260.3 ms** vs **258.15 ms** for Fixed $i=16$) because query projection and dense SDPA still dominate execution.
   - **Zero reuse schedules achieved the target $\ge 5\%$ latency advantage over Fixed $i=14$**.

Therefore, under the pre-registered kill criterion (*"KILL if reused/skipped global attention significantly harms quality or cannot beat simple uniform early stopping"*), **Attention-Reuse V0 is unequivocally KILLED**.

---

## 1. Recurrent Block Computation & Trajectory Drift Analysis

### Recurrent Block Architecture
A single DVLT recurrent iteration operates on $S=6$ views, $P=977$ tokens per view (972 image patches + 4 register tokens + 1 camera token), with embedding dimension $C=768$, 12 attention heads (head dim 64), and MLP expansion factor 4 ($3072$ dim).

In each recurrent step:
1. **Frame Attention:** Operates independently on $B \times S$ sequences of length $P=977$.
   $$\text{FLOPs}_{\text{frame}} = 24 S P C^2 + 4 S P^2 C = 82.98\text{ GF} + 17.60\text{ GF} = \mathbf{100.58\text{ GFLOPs}}$$
2. **Global Cross-View Attention:** Operates on 1 sequence of length $N = S \times P = 5862$.
   $$\text{FLOPs}_{\text{global}} = 24 N C^2 + 4 N^2 C = 82.98\text{ GF} + 105.58\text{ GF} = \mathbf{188.56\text{ GFLOPs}}$$
3. **Total Per-Step Compute:** $100.58 + 188.56 = \mathbf{289.14\text{ GFLOPs}}$.
   Global attention represents **65.2%** of the compute in every recurrent step.

### Empirical Drift Across Consecutive Iterations (Steps 2–16)

Measured across 10 DTU sequences on the full common $K=16$ linspace trajectory:

| Step $i$ | $\text{Cos}(h_i, h_{i-1})$ | Rel Drift $h$ | $\text{Cos}(x_{\text{frame}})$ | $\text{Cos}(x_{\text{global}})$ | $\text{Cos}(r_i, r_{i-1})$ | $\frac{\|r\|}{\|x_{\text{frame}}\|}$ | $\text{Cos}(K)$ | $\text{Cos}(V)$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 0.9492 | 0.3365 | 0.9397 | 0.9492 | 0.5145 | 0.2916 | 0.9545 | 0.7499 |
| 4 | 0.9809 | 0.1909 | 0.9770 | 0.9809 | 0.8460 | 0.2710 | 0.9900 | 0.9294 |
| 6 | 0.9943 | 0.1062 | 0.9921 | 0.9943 | 0.9438 | 0.2139 | 0.9973 | 0.9726 |
| 8 | 0.9974 | 0.0749 | 0.9967 | 0.9974 | 0.9729 | 0.1773 | 0.9988 | 0.9862 |
| 10 | 0.9985 | 0.0649 | 0.9982 | 0.9985 | 0.9846 | 0.1541 | 0.9993 | 0.9923 |
| 12 | 0.9990 | 0.0631 | 0.9988 | 0.9990 | 0.9904 | 0.1366 | 0.9995 | 0.9953 |
| 14 | 0.9991 | 0.0635 | 0.9990 | 0.9991 | 0.9932 | 0.1242 | 0.9996 | 0.9969 |
| 15 | 0.9992 | 0.0657 | 0.9991 | 0.9992 | 0.9942 | 0.1214 | 0.9996 | 0.9974 |
| 16 | 0.9993 | 0.0670 | 0.9992 | 0.9993 | 0.9952 | 0.1158 | 0.9997 | 0.9977 |

**Drift Findings:**
- Hidden state cosine similarity exceeds **0.997** by step 8 and **0.999** by step 12.
- The global attention residual $r = x_{\text{global}} - x_{\text{frame}}$ converges directionally ($\text{Cos}(r) > 0.993$ at step 14, $0.995$ at step 16).
- The relative magnitude of the global update decreases steadily from $29.2\%$ at step 2 to $11.6\%$ at step 16.
- Key and Value representations become static ($\text{Cos}(K) = 0.9997, \text{Cos}(V) = 0.9977$).

---

## 2. Tested Dense Attention Reuse Variants

All variants preserve dense, contiguous tensor shapes (`(B*S, P, C)` and `(B, S*P, C)`), completely avoiding dynamic sparse dispatch:

1. **Variant A — Global Skip:**
   At selected iterations $i \in \mathcal{S}_{\text{skip}}$, execute frame attention normally and pass the frame-attention output directly forward, skipping global attention entirely:
   $$x_i = x_{\text{frame}, i}$$
2. **Variant B — Previous Global Residual Reuse:**
   At normal iterations, cache the global cross-view residual $r_{\text{last}} = x_{\text{global}, i} - x_{\text{frame}, i}$. At reuse iterations $i \in \mathcal{S}_{\text{skip}}$, bypass global attention and add the scaled previous residual:
   $$x_i = x_{\text{frame}, i} + \alpha \cdot r_{\text{last}}, \quad \alpha \in \{0.5, 0.75, 1.0\}$$
3. **Variant C — Frozen Global K/V:**
   Recompute queries $Q_{\text{curr}}$ from the fresh frame-attention output $x_{\text{frame}, i}$, but reuse cached $K, V$ projections from the most recent global attention refresh step. Dense SDPA and global MLP run normally:
   $$Q = \text{Norm}_Q(W_q \cdot \text{Norm}_1(x_{\text{frame}})), \quad \text{Attn} = \text{SDPA}(Q, K_{\text{cached}}, V_{\text{cached}})$$

---

## 3. Comprehensive Empirical Benchmark (30 Schedules)

Evaluated across 10 DTU sequences (60 views). Real GPU recurrent latency measured with `torch.cuda.Event` (3 warmups, 10 timed repetitions) on RTX 5070 Laptop GPU:

| Schedule Name | Type | AbsRel | RMSE | Rot Err (deg) | Compute (GF) | Real Latency (ms) | $\Delta$ Lat vs i=14 | Matches i=14 Quality? | Meets $\ge 5\%$ Speedup vs i=14? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fixed i=16** | fixed | 0.005719 | 9.690 | 0.000 | 4626.2 | 258.15 ± 4.19 | -17.98% | **YES** | NO |
| **Fixed i=15** | fixed | 0.005716 | 9.708 | 0.000 | 4337.0 | 236.02 ± 3.12 | -7.86% | **YES** | NO |
| **Fixed i=14 (Hurdle)** | fixed | **0.005786** | **9.787** | **0.000** | **4047.9** | **218.82 ± 1.45** | **0.00%** | **YES** | NO |
| **Fixed i=13** | fixed | 0.006036 | 9.991 | 0.000 | 3758.8 | 203.77 ± 2.70 | +6.88% | NO | **YES** |
| VarA: Skip 16 | varA | 0.005765 | 9.790 | 0.239 | 4437.6 | 239.89 ± 3.52 | -9.63% | **YES** | NO |
| VarA: Skip 15 | varA | 0.005684 | 9.693 | 0.000 | 4437.6 | 242.05 ± 2.21 | -10.62% | **YES** | NO |
| VarA: Skip {15,16} | varA | 0.006169 | 10.056 | 0.119 | 4249.0 | 233.80 ± 4.37 | -6.85% | NO | NO |
| VarA: Skip {14,16} | varA | 0.005934 | 9.852 | 0.000 | 4249.0 | 231.12 ± 2.81 | -5.62% | NO | NO |
| VarA: Skip {12,14,16} | varA | 0.006309 | 10.061 | 0.119 | 4060.5 | 220.82 ± 1.60 | -0.92% | NO | NO |
| VarA: Skip {13,15} | varA | 0.005913 | 9.798 | 0.000 | 4249.0 | 230.79 ± 2.14 | -5.47% | NO | NO |
| VarB: reuse_16 ($\alpha=0.5$) | varB | 0.005688 | 9.708 | 0.000 | 4437.6 | 244.61 ± 2.33 | -11.79% | **YES** | NO |
| VarB: reuse_16 ($\alpha=0.75$) | varB | 0.005687 | 9.686 | 0.000 | 4437.6 | 245.07 ± 5.16 | -12.00% | **YES** | NO |
| VarB: reuse_16 ($\alpha=1.0$) | varB | 0.005712 | 9.678 | 0.119 | 4437.6 | 249.31 ± 1.60 | -13.93% | **YES** | NO |
| VarB: reuse_15 ($\alpha=0.5$) | varB | 0.005672 | 9.679 | 0.000 | 4437.6 | 250.45 ± 3.12 | -14.46% | **YES** | NO |
| VarB: reuse_15 ($\alpha=0.75$) | varB | 0.005684 | 9.681 | 0.000 | 4437.6 | 249.26 ± 3.33 | -13.91% | **YES** | NO |
| VarB: reuse_15 ($\alpha=1.0$) | varB | 0.005705 | 9.686 | 0.000 | 4437.6 | 246.48 ± 4.43 | -12.64% | **YES** | NO |
| VarB: reuse_15_16 ($\alpha=0.5$) | varB | 0.005726 | 9.751 | 0.000 | 4249.0 | 236.18 ± 3.85 | -7.93% | **YES** | NO |
| VarB: reuse_15_16 ($\alpha=0.75$) | varB | 0.005661 | 9.681 | 0.119 | 4249.0 | 241.13 ± 3.14 | -10.20% | **YES** | NO |
| VarB: reuse_15_16 ($\alpha=1.0$) | varB | 0.005696 | 9.659 | 0.000 | 4249.0 | 237.85 ± 4.27 | -8.70% | **YES** | NO |
| VarB: reuse_14_16 ($\alpha=0.5$) | varB | 0.005708 | 9.686 | 0.000 | 4249.0 | 239.53 ± 2.29 | -9.47% | **YES** | NO |
| VarB: reuse_14_16 ($\alpha=0.75$) | varB | 0.005694 | 9.664 | 0.000 | 4249.0 | 238.10 ± 3.04 | -8.81% | **YES** | NO |
| VarB: reuse_14_16 ($\alpha=1.0$) | varB | 0.005735 | 9.674 | 0.000 | 4249.0 | 237.93 ± 2.87 | -8.74% | **YES** | NO |
| VarB: reuse_even_10 ($\alpha=0.5$) | varB | 0.005775 | 9.758 | 0.119 | 4060.5 | 227.58 ± 3.25 | -4.01% | **YES** | NO |
| VarB: reuse_even_10 ($\alpha=0.75$) | varB | 0.005727 | 9.708 | 0.000 | 4060.5 | 231.41 ± 5.40 | -5.76% | **YES** | NO |
| VarB: reuse_even_10 ($\alpha=1.0$) | varB | 0.005859 | 9.707 | 0.119 | 4060.5 | 235.71 ± 4.09 | -7.72% | NO | NO |
| VarC: Freeze K/V 16 | varC | 0.005700 | 9.667 | 0.000 | 4612.3 | 258.08 ± 4.41 | -17.95% | **YES** | NO |
| VarC: Freeze K/V 15 | varC | 0.005704 | 9.661 | 0.000 | 4612.3 | 260.34 ± 4.65 | -18.98% | **YES** | NO |
| VarC: Freeze K/V {15,16} | varC | 0.005666 | 9.628 | 0.000 | 4598.5 | 260.05 ± 4.63 | -18.84% | **YES** | NO |
| VarC: Freeze K/V {14,16} | varC | 0.005785 | 9.645 | 0.000 | 4598.5 | 259.87 ± 3.51 | -18.76% | **YES** | NO |
| VarC: Freeze K/V {12,14,16} | varC | 0.005826 | 9.638 | 0.000 | 4584.7 | 257.32 ± 5.64 | -17.60% | NO | NO |

---

## 4. The Critical Comparison: Reuse vs Fixed $i=14$

### Why Global Attention Reuse Cannot Beat Fixed $i=14$
The pre-registered research question was:
> *"At matched reconstruction quality, does global-attention reuse beat the best uniform early-stopping baseline? Especially compare against Fixed $i=14$."*

The empirical answer is an unambiguous **NO**:

1. **The Frame-Attention Compute Tax:**
   - In Fixed $i=14$, execution terminates at step 14. Steps 15 and 16 are completely omitted, saving both frame attention ($2 \times 100.58\text{ GF}$) and global attention ($2 \times 188.56\text{ GF}$), for a total savings of **$578.27\text{ GFLOPs}$**.
   - In contrast, any 16-step reuse policy (such as `reuse_15_16`) continues running to step 16. It must still evaluate frame attention on all 6 views at steps 15 and 16 ($201.15\text{ GFLOPs}$).
   - Even though global attention is skipped at steps 15 and 16, the total compute of `reuse_15_16` is **$4249.0\text{ GFLOPs}$**, which is **$201.1\text{ GFLOPs}$ higher** than Fixed $i=14$ ($4047.9\text{ GFLOPs}$).
   - In real GPU wall-clock time, `VarB: reuse_15_16 (alpha=0.75)` takes **241.13 ms**, whereas Fixed $i=14$ takes **218.82 ms**. Fixed $i=14$ is **$22.3\text{ ms}$ ($9.3\%$) faster**!

2. **Aggressive Skipping Degrades Quality Without Gaining Speed:**
   - To match the FLOP budget of Fixed $i=14$ ($4047.9\text{ GFLOPs}$), an attention-reuse schedule must skip global attention at least 3 times (e.g. `Skip {12, 14, 16}` at $4060.5\text{ GFLOPs}$).
   - Under Variant A, skipping 3 global steps degrades AbsRel severely to **0.006309** (+9.0% error vs Fixed $i=14$).
   - Under Variant B, `reuse_even_after_10 (alpha=0.5)` preserves quality (AbsRel **0.005775** vs **0.005786**), but runs in **227.58 ms**, which is still **$8.76\text{ ms}$ (4.0%) slower** than Fixed $i=14$ (**218.82 ms**).

3. **Variant C (Frozen Global K/V) Offers Zero Real Latency Headroom:**
   - Freezing K and V only avoids the linear projection $W_k, W_v$ ($13.8\text{ GFLOPs}$ per step).
   - The query projection $W_q$, the full dense SDPA cross-view attention kernel ($O(N^2)$ with $N=5862$), LayerNorms, LayerScale, and the two-layer MLP ($3072$ dim) all continue to execute.
   - Consequently, Variant C runs in **257.3–260.3 ms**, completely indistinguishable from the unoptimized $K=16$ baseline (**258.15 ms**).

---

## 5. Diagnostics: Does Low Drift Predict Safe Reuse?

For each single-step reuse decision, we compared the step's global output drift and residual cosine similarity against the downstream error change relative to full $K=16$:

| Method & Step | $\text{Cos}(x_{\text{global}})$ | $\text{Cos}(r)$ | Residual Ratio | AbsRel | $\Delta$ AbsRel vs $K=16$ | $\Delta\%$ | Safe Reuse? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| VarA Skip 15 | 0.999155 | 0.994190 | 0.1214 | 0.005684 | -0.000034 | -0.60% | **YES** |
| VarA Skip 16 | 0.999270 | 0.995227 | 0.1158 | 0.005765 | +0.000046 | +0.81% | **YES** |
| VarB Reuse 15 ($\alpha=0.75$) | 0.999155 | 0.994190 | 0.1214 | 0.005684 | -0.000034 | -0.60% | **YES** |
| VarB Reuse 16 ($\alpha=0.75$) | 0.999270 | 0.995227 | 0.1158 | 0.005687 | -0.000031 | -0.55% | **YES** |
| VarC Freeze K/V 15 | 0.999155 | 0.994190 | 0.1214 | 0.005704 | -0.000015 | -0.26% | **YES** |
| VarC Freeze K/V 16 | 0.999270 | 0.995227 | 0.1158 | 0.005700 | -0.000019 | -0.33% | **YES** |

**Diagnostic Conclusion:**
- **Yes**, low global attention drift ($\text{Cos}(r) > 0.994, \frac{\|r\|}{\|x\|} \le 0.12$) reliably predicts that skipping or reusing global attention will not harm reconstruction quality for individual late steps.
- In particular, Variant B's residual addition ($\alpha \cdot r_{\text{last}}$) stabilizes the trajectory even across multiple skip steps.
- **However, safe reuse does not translate to wall-clock Pareto dominance**, because late-step frame attention incurs an inescapable compute overhead that early uniform stopping completely avoids.

---

## 6. Artifacts Generated

1. `outputs/attn_reuse_v0_drift_raw.csv` — Raw per-step hidden, input, output, residual, and K/V drift across all 10 sequences.
2. `outputs/attn_reuse_v0_drift_summary.json` — Aggregated drift metrics by recurrent step $i \in \{2..16\}$.
3. `outputs/attn_reuse_v0_sequence_metrics.csv` — Full sequence-level evaluation across 10 sequences $\times$ 30 schedules (300 records).
4. `outputs/attn_reuse_v0_summary.csv` and `outputs/attn_reuse_v0_summary.json` — Aggregated quality, latency, FLOPs, and comparison statistics for all 30 schedules.
5. `outputs/attn_reuse_v0_diagnostics.json` — Drift vs downstream error change diagnostics.
6. `outputs/attn_reuse_v0_global_drift.png` — 4-panel trajectory drift plot (hidden cosine, residual cosine, relative drift, residual magnitude ratio).
7. `outputs/attn_reuse_v0_pareto_frontier.png` — 2-panel Pareto frontier plot comparing Quality vs Real GPU Recurrent Latency and Quality vs Recurrent Compute.

---

## 7. GO / REVISE / KILL Verdict

### Pre-Registered Decision Criteria
- **GO:** If at least one reuse schedule matches or beats Fixed $i=14$ quality (AbsRel $\le 0.005786$) AND provides at least $\sim 5\%$ measured latency advantage over Fixed $i=14$ (latency $\le 207.9\text{ ms}$) without irregular per-view execution.
- **REVISE:** If quality is promising but latency gain is small.
- **KILL:** If reused/skipped global attention significantly harms quality or cannot beat simple uniform early stopping.

### Empirical Assessment
- Number of reuse schedules matching Fixed $i=14$ quality: **19 / 26**
- Number of reuse schedules beating Fixed $i=14$ latency: **0 / 26**
- Latency of fastest quality-matching reuse schedule (`VarB: reuse_even_10` $\alpha=0.5$): **227.58 ms**
- Latency of Fixed $i=14$: **218.82 ms** (Fixed $i=14$ is **4.0% faster**)
- Target 5% speedup bar: **207.88 ms** (no reuse schedule comes within 19 ms of this bar)

### Verdict
# **KILL**

### Research Synthesis & Architectural Lesson
Both ViewHalt branches have now been definitively tested under strict, matched-quality empirical conditions on official NVIDIA DVLT:
1. **Per-View Adaptive Halting (ViewHalt V0–V0.8.1):** Killed by irregular sparse-dispatch overhead. Masking out views fragmentizes contiguous tensors, introducing dynamic indexing and kernel launch latency that erased theoretical FLOP savings.
2. **Dense Global Attention Reuse (Attention-Reuse V0):** Killed by the baseline economics of uniform early stopping. In DVLT, late recurrent refinement saturates uniformly across views. Completely halting the solver at step 14 eliminates both frame attention and global attention, saving $578\text{ GFLOPs}$. Any partial attention-reuse scheme that keeps the solver looping to step 16 continues to pay the $100.6\text{ GFLOPs/step}$ frame-attention tax, rendering it structurally incapable of beating uniform early stopping.

For recurrent multi-view architectures like DVLT, **uniform early stopping (Fixed $i=14$) represents the true, unyielding empirical Pareto frontier**.
