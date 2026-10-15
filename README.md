# NeuroDash

**Detection-first glioma MRI analysis and visualization framework**

NeuroDash is a research-oriented framework for automated brain-tumour MRI analysis, visualization, segmentation review, 3D reconstruction, and AI-assisted reporting.

The framework was developed as part of MSc research at the **International Centre for Genetic Engineering and Biotechnology (ICGEB), New Delhi**, with the aim of providing a detection-first workflow for glioma sub-region analysis while supporting interactive visual inspection and reproducible evaluation.

> **Research software:** NeuroDash is intended for research and development purposes. It is not a clinically validated diagnostic system and must not be used as a substitute for assessment by qualified medical professionals.

---

## Overview

Conventional brain-tumour segmentation pipelines generally operate as dense 3D segmentation systems. NeuroDash explores a complementary **detection-first approach** in which tumour-bearing regions are first localized on axial MRI slices and subsequently segmented into tumour sub-regions.

The framework combines:

- Multi-parametric brain MRI preprocessing
- Axial tumour detection
- Tumour sub-region segmentation
- 3D reconstruction
- Interactive MRI visualization
- Interactive 3D visualization
- Rendering of volumetric views
- AI-assisted report generation
- A modular frontend/backend architecture

The software is designed to support research workflows in **medical image analysis, biomedical AI, computer vision, and glioma MRI segmentation**.

---

## NeuroDash Pipeline

The principal image-analysis workflow consists of the following stages:

```text
Multi-parametric MRI
        |
        v
Modality preparation
        |
        v
MRI intensity / gamma preprocessing
        |
        v
Axial slice construction
        |
        v
Tumour detection
        |
        v
Tumour-region segmentation
        |
        v
Coordinate / spatial reconstruction
        |
        v
3D volume reconstruction
        |
        v
Post-processing
        |
        v
Visualization and evaluation
---

## MRI Input and Preprocessing

NeuroDash is designed for multi-parametric brain MRI data containing the following commonly used sequences:

- T1-weighted MRI
- T1-contrast enhanced (T1ce / T1Gd)
- T2-weighted MRI
- FLAIR

For the principal RGB image representation used by the detection workflow:

```text
Red   = T1ce
Green = T2
Blue  = FLAIR
