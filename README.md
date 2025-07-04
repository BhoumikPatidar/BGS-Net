# Bottleneck‑Gated Segmentation Network (BGS‑Net)

**Efficient Conditional Decoder Activation for Semantic Segmentation of Satellite Imagery**

Satellite images often contain targets (e.g., buildings, roads, water bodies) that occupy only a small fraction of the scene. Standard encoder–decoder models like U‑Net perform full decoder computation on every patch, wasting resources when most patches are background. BGS‑Net introduces a lightweight gating mechanism on the encoder’s bottleneck features to decide—per patch—whether to run the costly decoder, cutting compute by over 56% with minimal accuracy loss.

---

## Key Features

- **Selective Decoder Activation**  
  A two‑stage gate analyzes bottleneck activations to skip decoder computation on non‑ROI patches, reducing decoder FLOPs by **56.62%**.

- **Zero Retraining Overhead**  
  Works with any pre‑trained U‑Net “as is”—no weight updates, no added parameters (overhead ≈ 0.0007% of one decoder conv).

- **Competitive Accuracy**  
  Building IoU of **0.6981** vs. **0.7145** for vanilla U‑Net (only a 2.3% relative drop focused on ROI patches).

- **Plug‑and‑Play Generality**  
  Easily applied to other sparse‑ROI tasks (medical lesions, crop mapping, road extraction).

---

## Problem Statement

1. **High Decoder Cost**  
   Decoder comprises ~65–70% of U‑Net’s FLOPs on every patch—even when most are background.

2. **Sparse Targets**  
   In many satellite scenes, buildings cover <15% of pixels, making uniform decoder use highly inefficient.

---

## Methodology

### 1. Bottleneck Feature Analysis

- **Bottleneck Tensor**  
  \(B \in \mathbb{R}^{C\times H'\times W'}\), where \(C=1024\), \(H'\times W'=32\times32\) for a 512×512 input.

- **Channel‑wise Mean**  
  \[
    \mu_c(B) = \frac{1}{H'W'} \sum_{i,j} B_{c,i,j}
  \]
  Reflects average activation per channel.

- **Discriminative Score**  
  \(\delta_c = \bigl|\mu^{\text{target}}_c - \mu^{\text{non‑target}}_c\bigr|\).  
  Channels with high \(\delta_c\) best separate ROI vs. background.

### 2. Two‑Stage Gating

1. **Stage 1: High‑Recall Detection**  
   - Select top \(K_1\) channels by \(\delta_c\).  
   - Threshold each channel via ROC‑derived \(\theta_c\):  
     \(g_c(B) = [\mu_c(B) > \theta_c]\).  
   - Combine with OR:  
     \(\displaystyle G_1(B) = \bigvee_{c\in\mathcal{C}_1} g_c(B)\).

2. **Stage 2: False‑Alarm Filtering**  
   - Select \(K_2\) channels sensitive to common confounders.  
   - Threshold via \(\gamma_c\):  
     \(f_c(B) = [\mu_c(B) > \gamma_c]\).  
   - Combine with AND:  
     \(\displaystyle G_2(B) = \bigwedge_{c\in\mathcal{C}_2} f_c(B)\).

3. **Final Gate**  
   \[
     G(B) = G_1(B) \;\wedge\; \neg G_2(B)
   \]  
   If \(G(B)=1\), run decoder; else output all‑background.

---

## Experimental Results

- **Dataset:** LandCoverAI (25 cm/pixel; classes: background, building, woodland, water, road).  
- **Baseline U‑Net:** Building IoU = 0.7145  
- **BGS‑Net:** Building IoU = 0.6981  
- **Decoder Execution Rate:** 43.4% of patches → 56.6% compute saved.  
- **Missed Cases:** Primarily very small buildings (~1.45% pixel coverage).

---

## Advantages

- **Compute Efficiency:** Halves decoder FLOPs with negligible gating cost.  
- **No Retraining:** Installs on pre‑trained models without weight changes.  
- **Adaptable:** Gate can be re‑analyzed for any new sparse‑ROI class.  
- **Complimentary:** Can be combined with pruning, quantization, or lightweight backbones.

---

## Applications

- **Medical Imaging:** Skip decoder on lesion‑free slices in MRI/CT scans.  
- **Agriculture:** Detect sparse disease or pest outbreaks in crop fields.  
- **Autonomous Driving:** Omit segmentation on empty road segments.  
- **Video Processing:** Leverage temporal coherence for frame‑level gating.

---

## Future Work

- **Multi‑Class Gating:** Joint gating for several target classes.  
- **Adaptive Thresholds:** Scene‑aware or dynamic gate parameters.  
- **Temporal Gating:** Incorporate frame history for video segmentation.  
- **Integration:** Merge with network compression and efficient backbone designs.

---

_By exploiting bottleneck activations already present in U‑Net, BGS‑Net makes sparse‑ROI segmentation practical at scale, delivering major compute savings with minimal impact on accuracy._  
