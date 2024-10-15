# Sentinel2ImageMatching
This project focuses on developing an algorithm to match and align Sentinel-2 satellite images
## Matching Examples

**SIFT-Brute-Force:**  
<img src="matches/BF_match_1.png" alt="SIFT-Brute-Force" width="400">

**SIFT-FLANN:**  
<img src="matches/FLANN_match_1.png" alt="SIFT-FLANN" width="400">

**RoMa:**  
<img src="matches/RoMa_match_1.png" alt="RoMa" width="400">

**Superglue:**  
<img src="matches/SuperGlue_match_1.png" alt="Superglue" width="400">


## Extraction Examples
**SIFT:**  
<img src="matches/SIFT_extraction_1.png" alt="SIFT" width="400">

**BRISK:**  
<img src="matches/BRISK_extraction_1.png" alt="BRISK" width="400">

**ORB:**  
<img src="matches/ORB_extraction_1.png" alt="ORB" width="400">

## Structure
- **algorithm.py**: Contains the core functionality for keypoint detection and image matching.
- **demo.ipynb**: A Jupyter notebook demonstrating the image matching algorithm in practice.
- **data**: Stores preprocessed satellite images from the [Deforestation in Ukraine dataset](https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine).
- **matches**: Holds examples of matched image pairs.


## Setup Environment

To set up the environment, run the following commands:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install .
