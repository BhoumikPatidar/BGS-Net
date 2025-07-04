# Bottleneck-Gated Segmentation Network (BGS‑Net)

Efficient Conditional Decoder Activation for Semantic Segmentation of Satellite Imagery

---

## Project Overview

Satellite images often contain targets (e.g., buildings, roads, water bodies) that occupy a small fraction of the scene. Standard encoder–decoder segmentation models like U‑Net perform full decoder computation on every patch, resulting in wasted resources when most patches lack targets.  
**BGS‑Net** introduces a lightweight gating mechanism that analyzes the bottleneck features of a pre‑trained U‑Net to decide whether to execute the costly decoder for each patch, achieving major computational savings with minimal accuracy loss :contentReference[oaicite:0]{index=0}.

---

## Key Contributions

- **Selective Decoder Activation**  
  Uses a two‑stage gating function on bottleneck feature statistics to skip decoder execution on non‑ROI patches, reducing compute by **56.62%** :contentReference[oaicite:1]{index=1}.  

- **Zero Retraining Overhead**  
  Works on any pre‑trained U‑Net without modifying weights or adding parameters; overhead is only ~0.0007% of a single decoder convolution :contentReference[oaicite:2]{index=2}.

- **Competitive Segmentation Quality**  
  Maintains a Building IoU of **0.6981**, versus **0.7145** for the vanilla U‑Net—a mere 2.3% relative drop focused on ROI patches :contentReference[oaicite:3]{index=3}.

- **Generalizable Framework**  
  Applicable to other sparse‑ROI tasks (medical scans, crop mapping, autonomous driving) by re‑analyzing bottleneck activations for new target classes :contentReference[oaicite:4]{index=4}.

---

## Problem Statement

1. **High Computation Cost**: Decoder accounts for ~65–70% of U‑Net’s FLOPs on every patch, even when most patches are background :contentReference[oaicite:5]{index=5}.  
2. **Sparse Targets**: In rural or suburban satellite scenes, buildings may cover <15% of pixels, leading to inefficiency in uniform processing :contentReference[oaicite:6]{index=6}.

---

## Methodology

### 1. Bottleneck Feature Analysis

- **Bottleneck Tensor**  
  Let **B** ∈ ℝ<sup>C×H′×W′</sup> be the output of the encoder bottleneck (C=1024 channels, H′×W′=32×32 for 512×512 input) :contentReference[oaicite:7]{index=7}.

- **Channel‑wise Mean Activation**  
  \\
  &nbsp;&nbsp;μ<sub>c</sub>(B) = (1/(H′·W′)) ∑<sub>i,j</sub> B<sub>c,i,j</sub>  
  Captures the average response per channel, reflecting presence of target features :contentReference[oaicite:8]{index=8}.

- **Discriminative Score (δ<sub>c</sub>)**  
  Difference between mean activations on target vs. non‑target patches:  
  δ<sub>c</sub> = |μ<sup>target</sup><sub>c</sub> – μ<sup>non‑target</sup><sub>c</sub>| :contentReference[oaicite:9]{index=9}.  
  Channels with highest δ<sub>c</sub> are most informative.

---

### 2. Two‑Stage Gating Mechanism

#### Stage 1: Target Detection (High Recall)
- **Select** K₁ channels with top δ<sub>c</sub> (e.g., channels 937, 117, 640 …)  
- **Thresholding** via ROC‑derived θ<sub>c</sub>:  
  gc(B,θ<sub>c</sub>) = 1 if μ<sub>c</sub>(B)>θ<sub>c</sub> (or <θ<sub>c</sub> for inverted channels)  
- **Combine** with logical OR to maximize recall:  
  G₁(B) = ∨<sub>i=1..K₁</sub> g<sub>cᵢ</sub>(B,θ<sub>cᵢ</sub>) :contentReference[oaicite:10]{index=10}.

#### Stage 2: False‑Alarm Filtering
- **Select** K₂ channels sensitive to common non‑target confounders (e.g., dense woodland)  
- **Thresholding** with γ<sub>c</sub> to detect false alarms:  
  f<sub>c</sub>(B,γ<sub>c</sub>) = 1 if μ<sub>c</sub>(B)>γ<sub>c</sub>  
- **Combine** with logical AND to filter only strong false alarms:  
  G₂(B) = ∧<sub>i=1..K₂</sub> f<sub>cᵢ</sub>(B,γ<sub>cᵢ</sub>) :contentReference[oaicite:11]{index=11}.

#### Final Gate
```text
G(B) = G₁(B) ∧ ¬G₂(B)
