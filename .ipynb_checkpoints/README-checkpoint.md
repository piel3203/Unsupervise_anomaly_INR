# Unsupervise anomaly detection using INR (Unsupervise_anomaly_INR)

## Description  
This repository contains the code associated with the work described in the article :
L. Piecuch, J. Huet, A. Frouin, A. Nordez, A. -S. Boureau and D. Mateus, "Unsupervised Anomaly Detection on Implicit Shape Representations for Sarcopenia Detection," 2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI), Houston, TX, USA, 2025, pp. 1-5, doi: 10.1109/ISBI60581.2025.10980714

**Sarcopenia** is a progressive age-related loss of muscle mass and strength that can significantly impacts patients daily life. Traditional assessment relies on 3D imaging and manual segmentations.  

This project analyzes **muscle shape** using an **implicit neural representation (INR)**. The method:  

- Models normal muscle shapes  
- Uses **unsupervised anomaly detection** to identify sarcopenic muscles from reconstruction errors  
- Learns a **latent representation** that separates normal and abnormal muscles directly form the labels  

Experiments on **103 segmented volumes** show that the method effectively discriminates sarcopenic and non-sarcopenic muscles, providing both **quantitative metrics** and **qualitative visualizations**.  

> **Clinical relevance:** This tool allows researchers and clinicians to detect sarcopenia are other muscle impacted shapes, directly from manual segmentation (from any modality, here the segmentation were created from 3D Ultrasound images), facilitating large-scale studies.  

![Overview](images_readme/general_idea_method.png)
---
## Table of Contents

1. [Architecture](#architecture)  
2. [Results](#results)  
   - [Qualitative](#qualitative-results)  
   - [Quantitative](#quantitative-results)  
3. [Installation](#installation)  
4. [Data](#data)  
5. [Usage](#usage)  
6. [References](#references)  

---
## Architecture

The model uses a **conditional INR with an auto-decoding strategy** to encode muscle shapes into a latent space and reconstruct them. Anomalies are detected based on reconstruction errors.  

![Architecture](images_readme/Unsupervised_anomaly_detection_method.png)  
*Figure: Overview of the INR model for muscle shape modeling.*  

---

## Results

### Qualitative Results

Example segmentation reconstructions highlighting normal vs. sarcopenic muscles:  

![Result 1](images_readme/qualitativ_results.png)  
*Figure: Qualitative results. Muscles from older adults with (a) and without (b) sarcopenia vs. from young subjects (c and d). Superposition of the GT (in green) and the prediction (in red)*  
> These overlays show how the model reconstructs muscle shapes. Larger reconstruction errors indicate sarcopenic muscles.  

Example of visualization of 2 first LDA components applied to the latent space of each subjects. 
![Result 2](images_readme/LDA.png) 
*Figure: Visualization of the first two LDA components applied to the INR’s latent representation of train and test participants for each muscle. 
Blue is for Y, dark green for OH and red for OS. A Fisher score and its corresponding p-value is calculated for each muscle.*

### Quantitative Results
Example of boxplots of Dice metric you can obtain for the 5 fold cross-validation for the 2 tests sets: Nomal shape (Y + OH) and Sarcopenic shape (OS) :
![Result 3](images_readme/quantitativ_results.png)  
*Figure: Box plot of Dice scores, from test predictions of healthy (Y+OH) and sarcopenic (OS) participants obtained with an INR model trained on healthy subjects, for the RF on different folds. The Dice score quantifies the volumetric overlap between the original segmented volume (manual ground truth) and the INR’s labelmap prediction. For each fold, healthy (Y+OH) adults were separated into train and test groups. OS were always in the test group. The p-value represents the overall effect of the OS group on Dice score, regardless of the fold.*

## Requirements & Installation  

### Prerequisites  

- Python 3.10.10 or higher  
- Common Python libraries for medical image processing: e.g. `numpy`, `scipy`, `nibabel`, `pandas`, `matplotlib`, possibly `torch` / `tensorflow` (depending on implementation)
- all the required libraries are in requirement.txt
- Optionally but recommanded: a CUDA-enabled GPU
  
### Install  

```bash
git clone https://github.com/piel3203/Unsupervise_anomaly_INR.git 
cd <Unsupervise_anomaly_INR>  
python -m venv venv  
source venv/bin/activate    # or `venv\Scripts\activate` on Windows  
pip install -r requirements.txt  
```

## Data 
The code requires **3D segmented muscle volumes** (no images other than the segmented labelmaps needed) for each subject. The code can also work on sparsen slices, but to capture the full shape differences we decided to use all the slices provided for the train and inference. A tradeoff can be done to manage the memory if needed by reducing the number of slices given as input. The segmentation labelmaps should be binary. If this is not the case, Please use the code ./ ... to binarize your labelmap if needed. 

- **Format**: 3D NIfTI files (.nii or .nii.gz) with integer labels for each muscle. Data can be directly converted from nrrd files with the small code provided in ... 

- **Subjects**: "Normal" (Y + OH) or Sarcopenic (OS).

- **Usage**: Place your volumes in the ./data folder. If you have multiple labelmaps for difference muscles, use this path architecture: ./data/muscle_name/labels . Filenames should be consistent with the subject IDs.

**Note**: The original clinical dataset is not publicly available. Use your own segmentations or request access from the corresponding institution.


## Usage 

## References


