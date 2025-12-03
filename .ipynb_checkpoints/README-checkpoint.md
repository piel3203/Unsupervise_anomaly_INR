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


## Requirements & Installation  

### Prerequisites  

- Python 3.x  
- Common Python libraries for medical image processing: e.g. `numpy`, `scipy`, `nibabel`, `pandas`, `matplotlib`, possibly `torch` / `tensorflow` (depending on implementation)  
- Optionally: a CUDA-enabled GPU — although the original study trained on a standard workstation, a GPU can accelerate training / inference. :contentReference[oaicite:9]{index=9}  

### Install  

```bash
git clone https://github.com/<your‑username>/<project‑repo>.git  
cd <project‑repo>  
python -m venv venv  
source venv/bin/activate    # or `venv\Scripts\activate` on Windows  
pip install -r requirements.txt  
