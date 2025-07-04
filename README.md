# Bottleneck‑Gated Segmentation Network (BGS‑Net)

**Efficient Conditional Decoder Activation for Semantic Segmentation of Satellite Imagery**

Satellite images often contain targets (e.g., buildings, roads, water bodies) that occupy only a small fraction of the scene. Standard encoder–decoder models like U‑Net perform full decoder computation on every patch, wasting resources when most patches are background. BGS‑Net introduces a lightweight gating mechanism on the encoder’s bottleneck features to decide—per patch—whether to run the costly decoder, cutting compute by over 56% with minimal accuracy loss.

---

## Key Features

- **Selective Decoder Activation**  
  A two‑stage gate analyzes bottleneck activations to skip decoder computation on non‑ROI patches, reducing decoder FLOPs by **56.62%**.

- **Zero Retraining Overhead**  
  Works with any pre‑trained U‑Net *as is*—no weight updates, no added parameters (overhead ≈ 0.0007% of one decoder conv).

- **Competitive Accuracy**  
  Building IoU of **0.6981** vs. **0.7145** for vanilla U‑Net (only a 2.3% relative drop focused on ROI patches).

- **Plug‑and‑Play Generality**  
  Easily applied to other sparse‑ROI tasks (medical lesions, crop mapping, road extraction).

--

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
- **Complementary:** Can be combined with pruning, quantization, or lightweight backbones.

---

## Applications

- **Medical Imaging:** Skip decoder on lesion‑free slices in MRI/CT scans.  
- **Agriculture:** Detect sparse disease or pest outbreaks in crop fields.  
- **Autonomous Driving:** Omit segmentation on empty road segments.  
- **Video Processing:** Leverage temporal coherence for frame‑level gating.

---

## Future Work

- **Multi‑Class Gating:** Joint gating for multiple target classes.  
- **Adaptive Thresholds:** Scene‑aware or dynamic gate parameters.  
- **Temporal Gating:** Incorporate frame history for video segmentation.  
- **Integration:** Merge with network compression and efficient backbone designs.

---

*Leveraging latent bottleneck activations, BGS‑Net enables efficient sparse‑ROI segmentation at scale, delivering major compute savings with minimal accuracy impact.*  
