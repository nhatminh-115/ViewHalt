# INTERVAL-ROBUST STAGE B: RECURRENT-DYNAMICS ADAPTATION WITH SHARED LoRA

**Status:** Formal Closure & Decision Record  
**Hypothesis Investigated:** Does randomized interval training require adaptation capacity inside the SHARED recurrent transformation itself (`LoopedAABlock` attention projections), rather than only inside its interval-gating MLP (`IntervalDepthScaling`)?  
**Scientific Decision:** **KILL** attention-LoRA interval adaptation.  
**Hardware & Safety:** Strictly capped at `torch.cuda.set_per_process_memory_fraction(0.85)` ($\le 6.9$ GB VRAM). Peak allocated VRAM: 6.22 GB (0 shared memory spillover, 0 OOMs).

---

## 1. Executive Summary & Decision Record

Stage A demonstrated that randomized interval training on `IntervalDepthScaling` alone yielded no meaningful advantage over matched Uniform FT. Stage B investigated whether this negative result stemmed from an adaptation bottleneck: namely, whether adapting the recurrent transformation itself via shared low-rank residuals (`LoRA`) unlocks interval-robustness.

| Configuration | Trainable Params | $K=12$ Uniform AbsRel | $K=12$ Non-Uniform Mean AbsRel | Treatment Effect ($\Delta \text{AbsRel}$) | Beats Pretrained $K=14$? |
|---|:---:|:---:|:---:|:---:|:---:|
| **Pretrained Baseline (Untouched)** | 0 | 0.005801 | 0.005844 | Baseline | Hurdle ($\mathbf{0.005713}$) |
| **Stage A: Gate Alone (Uniform FT)** | 316,032 | 0.005820 | 0.005822 | — | No |
| **Stage A: Gate Alone (Mild Rand FT)** | 316,032 | 0.005824 | 0.005829 | $-0.000007$ ($-0.1\%$) | No |
| **Stage B1: LoRA $r=4$ (Uniform FT)** | 352,896 | 0.005973 | 0.006029 | Control | No |
| **Stage B1: LoRA $r=4$ (Mild Rand FT)** | 352,896 | 0.005986 | 0.006023 | $\mathbf{+0.000006}$ ($\mathbf{+0.1\%}$) | No |
| **Stage B2: LoRA $r=8$ (Uniform FT)** | 389,760 | 0.006104 | 0.006168 | Control | No |
| **Stage B2: LoRA $r=8$ (Mild Rand FT)** | 389,760 | 0.006100 | 0.006154 | $\mathbf{+0.000014}$ ($\mathbf{+0.2\%}$) | No |

### Core Empirical Findings:
1. **Zero Treatment Advantage at Matched Capacity:**  
   Across both rank 4 (Stage B1) and rank 8 (Stage B2), the difference between matched Uniform FT and Mild Randomized FT is $\le 0.000014$ ($\le 0.2\%$), which is within random gradient variance. Randomized interval training provides no measurable benefit over uniform interval training when recurrent adaptation capacity is added.
2. **Failure of the Compute Hurdle:**  
   No model at $K=12$ comes close to matching the Pretrained $K=14$ benchmark ($\text{AbsRel} = 0.005713$). The best Stage B model at $K=12$ yields $0.005964$ ($+4.39\%$ worse).
3. **Overfitting / Capacity Degradation:**  
   Increasing adaptation capacity from rank 4 (36.9k LoRA params) to rank 8 (73.7k LoRA params) strictly *degrades* out-of-sample depth accuracy (from $+3.2\%$ error degradation at $r=4$ to $+5.3\%$ error degradation at $r=8$ relative to untouched pretrained).
4. **Inductive Robustness of Pretrained Recurrence:**  
   The untouched pretrained model is already exceptionally robust to severe interval perturbations: its AbsRel error increases by only $+0.88\%$ when schedule coefficient of variation $CV(\Delta t)$ escalates from $0.0$ to $0.6$. The robustness curves of Uniform FT and Randomized FT completely overlap with one another.

**Decision (Section 10 Compliance):**  
In accordance with Section 10 Decision Rules, both rank 4 and rank 8 have been rigorously evaluated and shown to yield no treatment advantage. We formally **KILL "attention-LoRA interval adaptation"**. Stage B3 (MLP LoRA) is rejected as unmotivated due to the negative capacity scaling observed in B2.

---

## 2. Recurrent Module & Capacity Audit

DVLT's recurrent backbone utilizes a shared `LoopedAABlock` iterated $K$ times. We audited the exact runtime module dimensions:

```
recurrent_blocks[0]:
  ├── frame_attn:
  │     ├── attn.qkv:   Linear(in_features=768, out_features=2304, bias=True)
  │     └── attn.proj:  Linear(in_features=768, out_features=768, bias=True)
  └── global_attn:
        ├── attn.qkv:   Linear(in_features=768, out_features=2304, bias=True)
        └── attn.proj:  Linear(in_features=768, out_features=768, bias=True)
```

### Parameter Ladder:
- **Pretrained Baseline:** 118,461,704 total parameters (0 trainable).
- **Stage A (`IntervalDepthScaling` alone):** 316,032 trainable parameters.
- **Stage B1 ($r=4$):**  
  - $316,032$ (gate) $+ 36,864$ (attention LoRA) = **$352,896$ trainable parameters**.  
  - Frame QKV ($768 \times 4 + 4 \times 2304 = 12,288$), Frame Proj ($768 \times 4 + 4 \times 768 = 6,144$).  
  - Global QKV ($12,288$), Global Proj ($6,144$).
- **Stage B2 ($r=8$):**  
  - $316,032$ (gate) $+ 73,728$ (attention LoRA) = **$389,760$ trainable parameters**.
- **Stage B3 ($r=8$ + MLP, contingent):**  
  - $316,032 + 73,728 + 122,880 = 512,640$ trainable parameters.

### Functional Zero-Drift Verification at Step 0:
LoRA linear layers were initialized with $A \sim \text{Kaiming Uniform}$ and $B = 0$:
$$W_{\text{eff}} = W_{\text{base}} + \frac{\alpha}{r} (x A) B$$
Numerically tested on sequence forward pass:
$$\max |\text{Depth}_{\text{pretrained}} - \text{Depth}_{\text{zero-LoRA}}| = 0.00000000\times 10^0$$
Exact mathematical identity to pretrained baseline at step 0 verified.

---

## 3. Learning-Rate Calibration

Calibrated on the matched Uniform FT control with `gate_lr = 1e-6` across $\text{lora\_lr} \in \{3\times 10^{-6}, 1\times 10^{-5}, 3\times 10^{-5}, 1\times 10^{-4}\}$:

| LoRA LR | Step 0 K=12 | Step 25 K=12 | Step 50 K=12 | Step 100 K=12 | Step 200 K=12 | Relative Drift @ Step 200 | $\le 5\%$ Stable? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$3.0\times 10^{-6}$** | 0.005801 | 0.005809 (+0.1%) | 0.005824 (+0.4%) | 0.005815 (+0.2%) | 0.005993 (+3.3%) | **$+3.31\%$** | **YES (Winner)** |
| **$1.0\times 10^{-5}$** | 0.005801 | 0.005840 (+0.7%) | 0.005888 (+1.5%) | 0.006002 (+3.5%) | 0.006601 (+13.8%) | $+13.80\%$ | NO |
| **$3.0\times 10^{-5}$** | 0.005801 | 0.006100 (+5.2%) | 0.006691 (+15.4%) | 0.006352 (+9.5%) | 0.006402 (+10.4%) | $+10.36\%$ | NO |
| **$1.0\times 10^{-4}$** | 0.005801 | 0.006898 (+18.9%) | 0.007129 (+22.9%) | 0.006275 (+8.2%) | 0.006167 (+6.3%) | $+6.32\%$ | NO |

**Conclusion:** $\text{lora\_lr} = 3.0\times 10^{-6}$ is the highest stable rate that maintains calibration within the strict $5\%$ drift tolerance throughout the entire 200-step trajectory. All higher rates destabilize the recurrent loop.

---

## 4. Matched Evaluation Results

All models were evaluated across 10 validation sequences (60 views) on 9 scene-disjoint DTU scans using official unnormalized benchmark preprocessing (`normalize_scene=False`).

### 4.1 Uniform Schedules ($K \in \{10, 12, 14, 16\}$)

| Model | $K=10$ AbsRel | $K=12$ AbsRel | $K=14$ AbsRel | $K=16$ AbsRel | Beats Pretrained $K=14$? |
|---|:---:|:---:|:---:|:---:|:---:|
| **Pretrained Baseline** | 0.006229 | 0.005801 | **0.005713** | 0.005719 | Reference Hurdle |
| **B1 Uniform FT ($r=4$)** | 0.006187 | 0.005973 | 0.005934 | 0.005926 | False |
| **B1 Mild Rand FT ($r=4$)** | 0.006144 | 0.005986 | 0.005923 | 0.005926 | False |
| **B2 Uniform FT ($r=8$)** | 0.006233 | 0.006104 | 0.006080 | 0.006050 | False |
| **B2 Mild Rand FT ($r=8$)** | 0.006235 | 0.006100 | 0.006065 | 0.006038 | False |

### 4.2 Non-Uniform Schedules ($K=12$)

| Schedule | Pretrained Baseline | B1 Uniform FT ($r=4$) | B1 Mild Rand FT ($r=4$) | B2 Uniform FT ($r=8$) | B2 Mild Rand FT ($r=8$) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Power $\gamma=0.75$** | 0.005866 | 0.006110 | 0.006077 | 0.006218 | 0.006215 |
| **Power $\gamma=1.25$** | 0.005864 | 0.005912 | 0.005923 | 0.006015 | 0.006005 |
| **Cosine Endpoints** | 0.005870 | 0.006096 | 0.006112 | 0.006255 | 0.006240 |
| **Best Random ($s=999$)** | 0.005822 | 0.006036 | 0.006043 | 0.006196 | 0.006170 |
| **Sampled Mild ($\sigma=0.2$)** | 0.005811 | 0.006004 | 0.005964 | 0.006134 | 0.006132 |
| **Sampled Moderate ($\sigma=0.5$)** | 0.005828 | 0.006015 | 0.006015 | 0.006189 | 0.006162 |
| **Mean Non-Uniform AbsRel** | **0.005844** | **0.006029** | **0.006023** | **0.006168** | **0.006154** |
| **Treatment Gain ($\Delta \text{AbsRel}$)** | — | Control | **$+0.000006$ (+0.1%)** | Control | **$+0.000014$ (+0.2%)** |

### 4.3 Robustness Curve: Error vs Perturbation $CV(\Delta t)$

| Schedule Perturbation $CV(\Delta t)$ | Pretrained Baseline | B1 Uniform FT ($r=4$) | B1 Mild Rand FT ($r=4$) | B2 Uniform FT ($r=8$) | B2 Mild Rand FT ($r=8$) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$CV = 0.0$** (Uniform) | 0.005801 | 0.005973 | 0.005986 | 0.006104 | 0.006100 |
| **$CV = 0.1$** | 0.005801 | 0.005989 | 0.005983 | 0.006115 | 0.006100 |
| **$CV = 0.2$** | 0.005814 | 0.005979 | 0.005979 | 0.006130 | 0.006111 |
| **$CV = 0.4$** | 0.005824 | 0.005995 | 0.005998 | 0.006123 | 0.006118 |
| **$CV = 0.6$** | 0.005852 | 0.006035 | 0.006031 | 0.006155 | 0.006145 |
| **Degradation ($CV=0.6$ vs $0.0$)** | **$+0.88\%$** | **$+1.04\%$** | **$+0.75\%$** | **$+0.84\%$** | **$+0.74\%$** |

---

## 5. Mechanistic Diagnostics

### 5.1 Parameter and Gate Drift:
- **Gate Drift ($\|s - s_0\|_2$):**  
  - B1 Uniform: $0.5897$ | B1 Mild: $0.5262$  
  - B2 Uniform: $0.5373$ | B2 Mild: $0.4744$
- **Per-Module LoRA Frobenius Norms:**  
  - `global_attn.attn.qkv`: **$0.129 - 0.135$** (highest adaptation norm across all arms).  
  - `frame_attn.attn.qkv`: **$0.115 - 0.120$**.  
  - `frame_attn.attn.proj`: **$0.073 - 0.079$**.  
  - `global_attn.attn.proj`: **$0.071 - 0.077$**.  
  Total LoRA update norm stabilized consistently at $\approx 0.20 - 0.21$.

### 5.2 Recurrent Hidden-State Drift ($K=16$ Uniform):
Measured by the relative Frobenius distance $\frac{\|h_{\text{model}}^{(i)} - h_{\text{pre}}^{(i)}\|_F}{\|h_{\text{pre}}^{(i)}\|_F}$ at recurrent iterations $i \in \{4, 8, 12, 14, 16\}$:

| Iteration | B1 Uniform FT ($r=4$) | B1 Mild Rand FT ($r=4$) | B2 Uniform FT ($r=8$) | B2 Mild Rand FT ($r=8$) |
|:---:|:---:|:---:|:---:|:---:|
| **Iteration 4** | 0.83% | 0.67% | 0.88% | 0.74% |
| **Iteration 8** | 1.38% | 1.24% | 1.56% | 1.46% |
| **Iteration 12** | 1.68% | 1.59% | 1.96% | 1.89% |
| **Iteration 14** | 1.74% | 1.66% | 2.03% | 1.97% |
| **Iteration 16** | 1.82% | 1.75% | 2.12% | 2.07% |

**Mechanistic Insight:**  
1. Cross-view global message passing (`global_attn.attn.qkv`) undergoes the largest adaptation weight updates.
2. Latent drift accumulates monotonically across the recurrent depth ($+0.7\% \to +2.1\%$).
3. Adapting the shared recurrent transformations does not refine coarse geometry or improve non-uniform step transitions; rather, each recurrent pass compounds residual divergence, which explains why higher capacity (rank 8) consistently achieves worse final depth accuracy than lower capacity (rank 4).

---

## 6. Interpretation Boundaries & Scientific Conclusion

1. **LoRA Framing:** Low-Rank Adaptation (LoRA) is an established parameter-efficient technique. It was utilized strictly as an experimental capacity probe to test whether limited recurrent plasticity enables randomized interval learning.
2. **Scientific Finding:**  
   The hypothesis that "randomized interval training requires adaptation capacity inside the shared recurrent transformation" is **empirically disproven**.
   - With 0 extra capacity (Pretrained), the architecture already exhibits near-perfect interval invariance ($+0.88\%$ error degradation under $CV=0.6$).
   - With gate-only capacity (Stage A: 316k params), treatment effect was $-0.1\%$.
   - With shared attention LoRA (Stage B1: 353k params, Stage B2: 390k params), treatment effect remains statistically indistinguishable from zero ($+0.1\%$ to $+0.2\%$).
3. **Formal Decision:**  
   **KILL** attention-LoRA interval adaptation. Close Stage B. Do not proceed to Stage B3 or full recurrent unfreezing.

---

## 7. Artifact Manifest

- `outputs/stageb_capacity_audit.json`: Audited runtime dimensions and parameter ladder.
- `outputs/stageb_lr_calibration.csv`: Learning rate calibration grid records.
- `outputs/stageb_matched_results.csv`: Matched evaluation results under Uniform $K \in \{10, 12, 14, 16\}$.
- `outputs/stageb_nonuniform_results.csv`: Matched evaluation across 7 non-uniform schedules at $K=12$.
- `outputs/stageb_robustness_curve.csv`: Empirical robustness curves ($CV \in [0.0, 0.6]$).
- `outputs/stageb_hidden_drift.csv`: Hidden-state drift across recurrent iterations $\{4, 8, 12, 14, 16\}$.
- `outputs/stageb_diagnostics.csv`: Per-module LoRA Frobenius norms, gate drift, and training latency.
- Visualizations:
  - `outputs/stageb_capacity_vs_gain.png`
  - `outputs/stageb_robustness_curves.png`
  - `outputs/stageb_quality_vs_k.png`
  - `outputs/stageb_hidden_state_drift.png`
