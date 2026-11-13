========================================================================
  Code of "Physical Computing Approaching the Data processing Inequality"
========================================================================

1. OVERVIEW
-----------
This repository contains the source code for our research on physical computing 
of radar target recognition. The project implements multiple recognition pipelines 
with similar network architechtures and parameters ranging from physical waveforms 
to 2D ISAR images. We focus on comparing the recognition performance across 
different transmit signals (LFM, Costas-LFM) and various proprocessing methods 
(Raw Echo, HRRP, ISAR).

2. FILE STRUCTURE & MODALITIES
------------------------------
Each script is self-contained and corresponds to a specific data processing 
and recognition methodology discussed in the paper.

- train_lfm.py
  Input: LFM Physical Waveforms.
  Model: Indentical 1D-CNN

- train_costas.py
  Input: Costas-LFM Physical Waveforms.
  Model: Indentical 1D-CNN

- train_abshrrp.py
  Input: HRRP profiles.
  Model: Indentical 1D-CNN

- train_ISAR.py
  Input: 2D Inverse Synthetic Aperture Radar images.
  Model: 2D-CNN with similar network architechtures and parameters

3. KEY TECHNICAL FEATURES
-------------------------
* Target-Level Cross-Validation: 
  To ensure rigorous evaluation and prevent data leakage, we implement a 
  target-level splitting strategy. Pulses belonging to the same target ID 
  are grouped together and assigned exclusively to either the training, 
  validation, or test set.

* Performance Metrics: 
  Each script performs K-fold cross-validation (10-fold for 1D signals, 
  5-fold for ISAR) and automatically generates accuracy.

* Data Export for MATLAB: 
  Results are exported as .csv files (e.g., CV_Matrix_...csv)


4. USAGE
--------
To reproduce the results for a specific signal type, run the corresponding 
script. For example, to train the model on raw LFM echoes:

Set train_lfm.py as file to run

Upon completion, a result directory named "NN-[Type]-[Timestamp]" will be 
created, containing the trained models (.h5) and full evaluation reports.

========================================================================