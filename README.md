# A Comparative Study of Unimodal and Multimodal IMU–sEMG Models for Knee Rehabilitation Exercise Quality Detection

This repository contains the complete experimental pipeline for unimodal (IMU-only and sEMG-only) and multimodal (fusion-based) deep learning models applied to the KneE-PAD dataset (https://doi.org/10.1038/s41597-025-04963-4). The project focuses on detecting correct and incorrect execution of knee rehabilitation exercises using wearable sensor data.

---

## Table of Contents
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Environment Setup](#environment-setup)
- [Quick Start](#quick-start)
- [Project Components](#project-components)
- [References](#references)

---

## Repository Structure

```
BEP/
├── eda.ipynb                                    # Exploratory data analysis
├── 1DCNN.py                                     # Unimodal 1D-CNN training (IMU or sEMG)
├── 1DCNN_eval.py                                # Unimodal model evaluation
├── fusion_models.py                             # Early, late, and hybrid fusion models
├── fusion_eval.py                               # Fusion model evaluation
├── error_analysis.py                            # Error-type analysis
├── metadata.txt                                 # Dataset metadata
├── dataset/                                     # KneE-PAD dataset (31 subjects)
│   ├── Subject_1/
│   ├── Subject_2/
│   └── ...Subject_31/
└── README.md                                    # This file
```

**Note**: The data preprocessing notebook (`KneE_PAD_Download_Read_and_Segment.ipynb`) is provided by the original KneE-PAD dataset authors and should be downloaded from the [official repository](https://github.com/ounospanas/KneE-PAD).

---

## Dataset

This project uses the publicly available **KneE-PAD dataset**, which contains synchronized IMU and sEMG recordings from 31 patients performing knee rehabilitation exercises with correct and incorrect execution patterns.

### Dataset Characteristics
- **Subjects**: 31 patients
- **Modalities**: Synchronized IMU (48 channels) and sEMG (8 channels)
- **Exercises**: 3 types (Squats, Leg Extensions, Gait)
- **Classes**: 9 (3 exercises × correct/incorrect variations)
### Download Instructions

1. Download the dataset and metadata from Zenodo:  
   https://doi.org/10.5281/zenodo.12112951

2. Extract the following files to the project root:
   - `dataset.zip` → Extract to `dataset/` folder
   - `metadata.txt` → Place in root directory

3. Download the preprocessing notebook from the official KneE-PAD repository:
   ```
   https://github.com/ounospanas/KneE-PAD/blob/main/KneE_PAD_Download_Read_and_Segment.ipynb
   ```

4. Run the preprocessing notebook:
   ```bash
   jupyter notebook KneE_PAD_Download_Read_and_Segment.ipynb
   ```
   This generates:
   - `emg_all.npy` - Preprocessed sEMG data 
   - `imu_all.npy` - Preprocessed IMU data
   - `labels_all.npy` - Class labels
   - `subjects_all.npy` - Subject IDs

---

## Environment Setup

### Requirements
- Python 3.8+
- TensorFlow 2.10+
- scikit-learn
- NumPy
- Matplotlib
- Jupyter Notebook

### Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy tensorflow scikit-learn matplotlib jupyter pandas seaborn
```

---

## Quick Start

### 1. Data Preprocessing
Download the preprocessing notebook from the [official KneE-PAD repository](https://github.com/ounospanas/KneE-PAD) and run it:
```bash
jupyter notebook KneE_PAD_Download_Read_and_Segment.ipynb
```
This notebook handles dataset download and preprocessing to generate the required `.npy` files.

### 2. Exploratory Data Analysis
```bash
jupyter notebook eda.ipynb
```

### 3. Train Unimodal Models

**Train IMU-only 1D-CNN:**
```bash
# Modify 1DCNN.py line 17 to use IMU data:
# X = imu
# modality = "imu"
python 1DCNN.py
```

**Train sEMG-only 1D-CNN:**
```bash
# Modify 1DCNN.py line 17 to use sEMG data:
# X = emg
# modality = "emg"
python 1DCNN.py
```

### 4. Train Fusion Models
```bash
python fusion_models.py
```
This script trains three fusion architectures:
- **Early Fusion**: Concatenate raw IMU and sEMG data before feature extraction
- **Late Fusion**: Extract features separately then concatenate before classification
- **Hybrid Fusion**: Combine early and late fusion strategies

### 5. Evaluate Models
```bash
# Unimodal evaluation
python 1DCNN_eval.py

# Fusion model evaluation
python fusion_eval.py

# Error analysis with bootstrap confidence intervals
python error_analysis.py
```

---

## Project Components

### 1. **Data Preprocessing** 
The preprocessing notebook (`KneE_PAD_Download_Read_and_Segment.ipynb`) is provided by the original dataset authors and can be found at:
- [KneE-PAD GitHub Repository](https://github.com/ounospanas/KneE-PAD/blob/main/KneE_PAD_Download_Read_and_Segment.ipynb)

This notebook:
- Downloads KneE-PAD dataset from Zenodo
- Segments raw sensor data into individual exercises
- Applies z-score normalization per segment
- Generates NumPy arrays for training

### 2. **Exploratory Data Analysis** (`eda.ipynb`)
- Visualizes IMU and sEMG data distributions
- Analyzes temporal characteristics
- Class balance assessment
- Subject-level variations

### 3. **Unimodal Models** (`1DCNN.py`)
**Architecture:**
- Input: Variable-length 1D temporal sequences
- Conv1D layers with BatchNormalization and ReLU
- GlobalAveragePooling1D for temporal compression
- Dense layers with Dropout for classification

**Key Features:**
- Z-score normalization (per-segment, per-channel)
- 70-15-15 train-validation-test split with stratification
- Reproducible random seeds for consistency

### 4. **Fusion Models** (`fusion_models.py`)
Implements three fusion strategies:

1. **Early Fusion**: Concatenate raw modalities → shared backbone
2. **Late Fusion**: Separate backbones → concatenate before classification  
3. **Hybrid Fusion**: Combine early and late fusion paths

All use the same base Conv1D architecture with batch normalization.

### 5. **Model Evaluation** (`1DCNN_eval.py`, `fusion_eval.py`)
- Accuracy, Precision, Recall, F1-score
- Confusion matrices
- Per-class performance metrics
- Classification reports

### 6. **Error Analysis** (`error_analysis.py`)
- Per-class F1-score computation
- Bootstrap confidence intervals (95% CI)
- Error type classification
- Misclassification pattern analysis

---

## References

1. **KneE-PAD Dataset**: [https://doi.org/10.1038/s41597-025-04963-4](https://doi.org/10.1038/s41597-025-04963-4)
2. **Dataset on Zenodo**: [https://doi.org/10.5281/zenodo.12112951](https://doi.org/10.5281/zenodo.12112951)

