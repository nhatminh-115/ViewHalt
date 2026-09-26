# INTERVAL-ROBUST STAGE B.1: MATCHED-CONTROL & CHECKPOINT REPAIR REPORT

**Status:** Formal Closure & Decision Record  
**Hypothesis Tested:** Does randomized interval training provide a meaningful advantage over matched uniform interval training when evaluated under strictly decoupled RNG streams, matched step-0 initialization, independent rank-8 LR calibration, and checkpoint-wise comparison?  
**Scientific Decision:** **ATTENTION-LoRA STAGE B KILL**  
**Hardware & Safety:** Capped at `torch.cuda.set_per_process_memory_fraction(0.85)` ($\le 6.9$ GB VRAM). Peak allocated VRAM: 6.22 GB (0 shared memory spillover, 0 OOMs).

---

## 1. Executive Summary & Core Protocol Repairs

Stage B.1 addressed three critical protocol concerns identified in the initial Stage-B evaluation:
1. **RNG Stream Decoupling**: In Stage B, the data sampling and interval sampling shared a single NumPy RNG, causing the data stream to diverge between Uniform and Randomized arms after step 0. Stage B.1 pre-generated an immutable training manifest ([`outputs/stageb1_training_manifest.csv`](file:///d:/Study/ViewHalt/outputs/stageb1_training_manifest.csv)) fixing the exact scene sequence, view/sample seeds, and sampled $K$ for all 200 steps. Uniform and Randomized arms were asserted to receive 100% byte-identical data streams.
2. **Independent Rank-8 LR Calibration**: Rank 8 was independently calibrated on the Uniform control across $\text{lora\_lr} \in \{5\times 10^{-7}, 1\times 10^{-6}, 3\times 10^{-6}, 1\times 10^{-5}\}$, identifying $\mathbf{1.0\times 10^{-6}}$ as the optimal stable rate (achieving $+1.53\%$ drift at step 200, whereas $3\times 10^{-6}$ reached $+5.96\%$).
3. **Checkpoint-Wise Evaluation**: Rather than evaluating only at step 200 (where mild over-training occurs), models were evaluated across checkpoints $\text{step} \in \{0, 25, 50, 100, 200\}$ over all 12 schedules (Uniform $K \in \{12, 14\}$, 6 non-uniform schedules, and $CV \in \{0.0, 0.2, 0.4, 0.6\}$).

---

## 2. Checkpoint-Wise Matched Comparison Table

All metrics below are computed over the 10 official validation sequences (60 views) on scene-disjoint DTU scans using standard unnormalized benchmark preprocessing.

| Rank | LoRA LR | Step | Uniform Control $K=12$ | Mild Rand FT $K=12$ | $K=12$ Treatment Gain ($\Delta \text{AbsRel}$) | Uniform Control Non-Unif Mean | Mild Rand Non-Unif Mean | Non-Unif Treatment Gain ($\Delta \text{AbsRel}$) | Pretrained $K=14$ Hurdle | Rand Beats Hurdle? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| — | — | **Pretrained** | **0.005801** | **0.005801** | **0.0** | **0.005844** | **0.005844** | **0.0** | **0.005713** | Baseline |
| **4** | $3\times 10^{-6}$ | 0 | 0.005801 | 0.005801 | 0.0 | 0.005844 | 0.005844 | 0.0 | 0.005713 | False |
| **4** | $3\times 10^{-6}$ | 25 | 0.005806 | 0.005803 | $+0.000003$ ($+0.06\%$) | 0.005848 | 0.005849 | $-0.000001$ ($-0.02\%$) | 0.005713 | False |
| **4** | $3\times 10^{-6}$ | **50** | **0.005811** | **0.005806** | $\mathbf{+0.000005}$ ($\mathbf{+0.09\%}$) | **0.005863** | **0.005855** | $\mathbf{+0.000009}$ ($\mathbf{+0.15\%}$) | 0.005713 | False |
| **4** | $3\times 10^{-6}$ | 100 | 0.005847 | 0.005859 | $-0.000013$ ($-0.22\%$) | 0.005891 | 0.005892 | $-0.000001$ ($-0.01\%$) | 0.005713 | False |
| **4** | $3\times 10^{-6}$ | 200 | 0.006048 | 0.006060 | $-0.000012$ ($-0.20\%$) | 0.006111 | 0.006118 | $-0.000008$ ($-0.13\%$) | 0.005713 | False |
| **8** | $1\times 10^{-6}$ | 0 | 0.005801 | 0.005801 | 0.0 | 0.005844 | 0.005844 | 0.0 | 0.005713 | False |
| **8** | $1\times 10^{-6}$ | 25 | 0.005801 | 0.005814 | $-0.000013$ ($-0.22\%$) | 0.005849 | 0.005848 | $+0.000001$ ($+0.02\%$) | 0.005713 | False |
| **8** | $1\times 10^{-6}$ | **50** | **0.005802** | **0.005787** | $\mathbf{+0.000015}$ ($\mathbf{+0.25\%}$) | **0.005855** | **0.005851** | $\mathbf{+0.000004}$ ($\mathbf{+0.08\%}$) | 0.005713 | False |
| **8** | $1\times 10^{-6}$ | 100 | 0.005834 | 0.005826 | $+0.000008$ ($+0.14\%$) | 0.005867 | 0.005855 | $+0.000011$ ($+0.19\%$) | 0.005713 | False |
| **8** | $1\times 10^{-6}$ | 200 | 0.005867 | 0.005887 | $-0.000020$ ($-0.35\%$) | 0.005930 | 0.005928 | $+0.000002$ ($+0.03\%$) | 0.005713 | False |

*Note: Treatment Gain is defined as $\text{AbsRel}_{\text{Uniform}} - \text{AbsRel}_{\text{Rand}}$ (positive values indicate Randomized FT achieves lower error).*

---

## 3. Analysis of Primary Scientific Questions

### Question A & B: Consistency of Treatment Effect on Uniform and Non-Uniform $K=12$
- **Rank 4**:
  - Across all training steps $\{25, 50, 100, 200\}$, the treatment effect fluctuates between $-0.000013$ and $+0.000005$ on Uniform $K=12$, and between $-0.000008$ and $+0.000009$ on the non-uniform mean.
  - At the optimal checkpoint (Step 50), Mild Rand FT is $0.005855$ vs Uniform Control $0.005863$ ($\Delta = +0.000009$, or $+0.15\%$).
- **Rank 8 (with independent calibration)**:
  - Across all checkpoints, the treatment effect fluctuates between $-0.000020$ and $+0.000015$ on Uniform $K=12$, and between $+0.000001$ and $+0.000011$ on the non-uniform mean.
  - At Step 50, Mild Rand FT is $0.005787$ vs Uniform Control $0.005802$ ($\Delta = +0.000015$, or $+0.25\%$).
- **Conclusion**: Across both ranks and across all checkpoints, no practically meaningful treatment signal was observed at seed 42. Differences are confined within $\pm 0.000020$ ($\pm 0.3\%$), showing no systematic or meaningful divergence between uniform and randomized interval fine-tuning.

### Question C: Schedule-by-Schedule Performance
In `outputs/stageb1_nonuniform_by_checkpoint.csv`:
- **Power $\gamma=0.75$**: At Step 50, Rank 4 diff is $+0.000002$; Rank 8 diff is $-0.000013$.
- **Power $\gamma=1.25$**: At Step 50, Rank 4 diff is $+0.000025$; Rank 8 diff is $+0.000008$.
- **Cosine Endpoints**: At Step 50, Rank 4 diff is $+0.000025$; Rank 8 diff is $+0.000007$.
- **Best Random ($s=999$)**: At Step 50, Rank 4 diff is $+0.000005$; Rank 8 diff is $-0.000013$.
- **Sampled Mild ($\sigma=0.2$)**: At Step 50, Rank 4 diff is $-0.000002$; Rank 8 diff is $+0.000008$.
- **Sampled Moderate ($\sigma=0.5$)**: At Step 50, Rank 4 diff is $-0.000003$; Rank 8 diff is $+0.000008$.
There is no schedule where Randomized FT demonstrates a consistent or substantial advantage over matched Uniform FT.

### Question D: Robustness Curves Across Perturbations $CV(\Delta t)$
In `outputs/stageb1_robustness_by_checkpoint.csv`:
- At Step 50 for Rank 8:
  - $CV = 0.0$: Uniform = $0.005802$, Rand = $0.005787$
  - $CV = 0.2$: Uniform = $0.005815$, Rand = $0.005834$
  - $CV = 0.4$: Uniform = $0.005844$, Rand = $0.005834$
  - $CV = 0.6$: Uniform = $0.005857$, Rand = $0.005824$
- The slope of degradation from $CV=0.0$ to $CV=0.6$ is $+0.95\%$ for Uniform Control and $+0.64\%$ for Randomized FT. Both curves remain nearly identical to the untouched Pretrained baseline ($+0.88\%$).

### Question E: Pretrained Uniform $K=14$ Compute Hurdle ($0.005713$)
- **Hurdle Benchmark**: Pretrained Uniform $K=14$ achieves $\text{AbsRel} = \mathbf{0.005713}$.
- Best observed $K=12$ configuration across all ranks, checkpoints, and arms:
  - Rank-8 Mild Rand FT at Step 50: $\mathbf{0.005787}$ ($+1.30\%$ worse than hurdle).
  - Rank-8 Uniform Control at Step 50: $\mathbf{0.005802}$ ($+1.56\%$ worse than hurdle).
- **Result**: No randomized-trained $K=12$ configuration reaches or beats the Pretrained $K=14$ hurdle.

---

## 4. Scientific Decision (Section 8 Compliance)

Per Section 8 of the protocol:
> "ATTENTION-LoRA STAGE B KILL: After identical data trajectory, matched initialization, checkpoint-wise comparison, and independent rank-4/rank-8 calibration, neither rank shows a meaningful Randomized-vs-Uniform advantage."

**Formal Determination:**
1. Under 100% byte-identical data manifests, matched initialization, and independent rank-8 LR calibration, no practically meaningful treatment signal was observed at seed 42 at any checkpoint between step 0 and step 200.
2. In accordance with Section 8, we issue a formal **KILL on Attention-LoRA Interval Adaptation**.
3. In accordance with Section 11, multi-seed compute (seeds 43, 44) is omitted as no positive treatment signal was detected.
4. Stage B is formally and comprehensively closed.

---

## 5. Artifact Manifest

- `outputs/stageb1_training_manifest.csv`: 200-step immutable data manifest with fixed scenes, seeds, and pre-sampled $K$.
- `outputs/stageb1_rank8_lr_calibration.csv`: Independent Rank-8 LR calibration records across $\{5\times 10^{-7}, 1\times 10^{-6}, 3\times 10^{-6}, 1\times 10^{-5}\}$.
- `outputs/stageb1_checkpoint_matched.csv`: Matched checkpoint-wise evaluation table across steps $\{0, 25, 50, 100, 200\}$.
- `outputs/stageb1_nonuniform_by_checkpoint.csv`: Full breakdown across 6 non-uniform schedules at each checkpoint.
- `outputs/stageb1_robustness_by_checkpoint.csv`: Full robustness curve data ($CV \in [0.0, 0.6]$) across checkpoints.
- Visualizations:
  - `outputs/stageb1_treatment_effect_vs_step.png`
  - `outputs/stageb1_quality_vs_step.png`
  - `outputs/stageb1_robustness_at_checkpoints.png`
