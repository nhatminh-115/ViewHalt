# STAGE B.1C: FINAL 50-STEP OFFICIAL-K CONFIRMATION REPORT

**Status:** Final Scientific Closure & Irrevocable Decision  
**Hypothesis Tested:** Does randomized interval training with shared attention LoRA demonstrate a consistent $\ge 0.5\%$ advantage over matched uniform fine-tuning when trained with official DVLT $\text{Beta}(2,1)$ K-sampling at the historically most favorable checkpoint (Step 50)?  
**Scientific Decision:** **FINAL KILL — Attention-LoRA Interval Adaptation** (Do not revisit Stage B).  
**Hardware & Safety:** Capped at `torch.cuda.set_per_process_memory_fraction(0.85)` ($\le 6.9$ GB VRAM). Peak allocated VRAM: 6.22 GB (0 shared memory spillover, 0 OOMs).

---

## 1. Executive Summary & Verification of Corrected K-Sampling

This final confirmatory run resolved the K-sampler regression in the training manifest by replacing the previously used symmetric $\text{Beta}(2,2)$ distribution with the exact native DVLT runtime implementation of $\text{Beta}(2,1)$.

### 1.1 Runtime Parameter Verification:
Confirmed directly from the instantiated model runtime:
- `min_steps`: $8$
- `num_steps`: $16$
- `k_sampler_beta_a`: $2$
- `k_sampler_beta_b`: $1$

### 1.2 Manifest Comparison & Distribution:
- **Scene, Video Index, and View Trajectory**: Preserved 100% identically from the Stage B.1 manifest across all steps.
- **Old Manifest Mean $K$ ($\text{Beta}(2,2)$)**: $11.9800$
- **Corrected Official Manifest Mean $K$ ($\text{Beta}(2,1)$)**: $\mathbf{13.5400}$
- **Histogram of Sampled $K$ (200 steps)**:
  - $K=9$: $8$ ($4.0\%$)
  - $K=10$: $8$ ($4.0\%$)
  - $K=11$: $15$ ($7.5\%$)
  - $K=12$: $22$ ($11.0\%$)
  - $K=13$: $32$ ($16.0\%$)
  - $K=14$: $41$ ($20.5\%$)
  - $K=15$: $47$ ($23.5\%$)
  - $K=16$: $27$ ($13.5\%$)
- Saved to [`outputs/stageb1c_official_k_manifest.csv`](file:///d:/Study/ViewHalt/outputs/stageb1c_official_k_manifest.csv).
- Data trajectory equivalence between Uniform Control and Randomized FT was asserted and confirmed **100% byte-identical** for both Rank 4 and Rank 8.

---

## 2. Empirical Results at Step 50

Evaluation conducted across 10 validation sequences (60 views) on scene-disjoint DTU scans with standard unnormalized benchmark preprocessing.

### 2.1 Summary Comparison Table (Step 50):

| Rank | Schedule Key | Schedule Description | Pretrained Baseline | Uniform Control AbsRel | Mild Rand FT AbsRel | Treatment Gain ($\Delta \text{AbsRel}$) | Treatment Gain (%) | Beats Pretrained $K=14$ Hurdle? |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| — | `uniform_K14` | Pretrained $K=14$ Reference | **0.005713** | — | — | — | — | Hurdle Benchmark |
| **4** | `uniform_K12` | Uniform $K=12$ | 0.005801 | 0.005793 | 0.005807 | $-0.000014$ | $-0.25\%$ | False |
| **4** | `uniform_K14` | Uniform $K=14$ | 0.005713 | 0.005761 | 0.005743 | $+0.000018$ | $+0.31\%$ | False |
| **4** | `nonunif_mean`| Non-Uniform $K=12$ Mean (6 sched.) | 0.005844 | 0.005857 | 0.005849 | $\mathbf{+0.000008}$ | $\mathbf{+0.14\%}$ | False |
| **4** | `robust_cv_0.6`| Robustness $CV=0.601$ | 0.005852 | 0.005840 | 0.005837 | $+0.000003$ | $+0.05\%$ | False |
| **8** | `uniform_K12` | Uniform $K=12$ | 0.005801 | 0.005806 | 0.005818 | $-0.000012$ | $-0.20\%$ | False |
| **8** | `uniform_K14` | Uniform $K=14$ | 0.005713 | 0.005723 | 0.005722 | $+0.000001$ | $+0.01\%$ | False |
| **8** | `nonunif_mean`| Non-Uniform $K=12$ Mean (6 sched.) | 0.005844 | 0.005848 | 0.005852 | $\mathbf{-0.000004}$ | $\mathbf{-0.07\%}$ | False |
| **8** | `robust_cv_0.6`| Robustness $CV=0.601$ | 0.005852 | 0.005839 | 0.005837 | $+0.000002$ | $+0.03\%$ | False |

*Treatment Gain is defined as $\text{AbsRel}_{\text{Uniform}} - \text{AbsRel}_{\text{Rand}}$ (positive values indicate Randomized FT achieves lower error).*

### 2.2 Detailed Non-Uniform Schedule Breakdown:

| Rank | Schedule Key | Pretrained Baseline | Uniform Control | Mild Rand FT | Treatment Gain ($\Delta \text{AbsRel}$) | Treatment Gain (%) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **4** | Power $\gamma=0.75$ | 0.005866 | 0.005893 | 0.005887 | $+0.000006$ | $+0.11\%$ |
| **4** | Power $\gamma=1.25$ | 0.005864 | 0.005827 | 0.005817 | $+0.000010$ | $+0.17\%$ |
| **4** | Cosine endpoints | 0.005870 | 0.005882 | 0.005879 | $+0.000003$ | $+0.05\%$ |
| **4** | Best random ($s=999$) | 0.005822 | 0.005866 | 0.005855 | $+0.000011$ | $+0.19\%$ |
| **4** | Sampled mild ($\sigma=0.2$) | 0.005811 | 0.005823 | 0.005820 | $+0.000003$ | $+0.06\%$ |
| **4** | Sampled moderate ($\sigma=0.5$) | 0.005828 | 0.005855 | 0.005840 | $+0.000015$ | $+0.25\%$ |
| **8** | Power $\gamma=0.75$ | 0.005866 | 0.005887 | 0.005901 | $-0.000014$ | $-0.23\%$ |
| **8** | Power $\gamma=1.25$ | 0.005864 | 0.005838 | 0.005840 | $-0.000003$ | $-0.05\%$ |
| **8** | Cosine endpoints | 0.005870 | 0.005864 | 0.005882 | $-0.000018$ | $-0.31\%$ |
| **8** | Best random ($s=999$) | 0.005822 | 0.005840 | 0.005849 | $-0.000009$ | $-0.15\%$ |
| **8** | Sampled mild ($\sigma=0.2$) | 0.005811 | 0.005813 | 0.005808 | $+0.000005$ | $+0.09\%$ |
| **8** | Sampled moderate ($\sigma=0.5$) | 0.005828 | 0.005847 | 0.005833 | $+0.000014$ | $+0.23\%$ |

---

## 3. Evaluation Against Decision Rules

The decision criteria defined in Section 6 state:
> - If Randomized FT improvement over matched Uniform FT remains below $\sim 0.5\%$ and is not consistent across schedules: **FINAL KILL attention-LoRA interval adaptation. Do not revisit Stage B.**
> - If $\ge 0.5\%$ consistent treatment advantage appears across multiple schedules: **REOPEN only for multi-seed verification.**

### Quantitative Assessment:
1. **Magnitude of Treatment Gain**:
   - For Rank 4, the mean non-uniform treatment gain is **$+0.14\%$** ($+0.000008$), with a maximum across any schedule of $+0.25\%$.
   - For Rank 8, the mean non-uniform treatment gain is **$-0.07\%$** (negative: Uniform Control is superior), with 4 out of 6 schedules favoring Uniform Control.
   - On standard Uniform $K=12$, the treatment effect is negative for both ranks ($-0.25\%$ for Rank 4, $-0.20\%$ for Rank 8).
   - In no configuration does the treatment effect approach or exceed the $+0.5\%$ decision threshold.
2. **Compute Hurdle ($K=14$ Benchmark $= 0.005713$)**:
   - The lowest $K=12$ error achieved by any model is $0.005793$ (Rank 4 Uniform Control) and $0.005806$ (Rank 8 Uniform Control).
   - No randomized-trained $K=12$ model reaches the Pretrained $K=14$ hurdle ($+1.6\%$ gap).
3. **Consistency**:
   - Treatment gains are inconsistent across schedules and ranks, fluctuating within stochastic gradient noise ($\pm 0.000018$, or $\pm 0.3\%$).

---

## 4. Final Scientific Decision

1. **FINAL KILL**: Under official $\text{Beta}(2,1)$ K-sampling at the best-case Step 50 checkpoint, randomized interval training yields no practically meaningful advantage over matched uniform fine-tuning ($< 0.5\%$ across all schedules).
2. **Stage B Closure**: Stage B (recurrent attention LoRA adaptation) is definitively and irrevocably closed.
3. **Future Work**: Do not revisit Stage B or proceed to MLP LoRA rescues within this evaluation line.

---

## 5. Artifact Manifest

- `outputs/stageb1c_official_k_manifest.csv`: 200-step training manifest with official $\text{Beta}(2,1)$ K-sampling.
- `outputs/stageb1c_results.csv`: Step 50 evaluation results across all 9 schedules for Rank 4 and Rank 8.
