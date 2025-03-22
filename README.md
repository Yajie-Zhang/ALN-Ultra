# ALN-Ultra

Breast axillary lymph node (ALN) metastasis is a critical determinant of treatment strategies and patient prognosis in early-stage breast cancer. While ultrasound imaging has emerged as a non-invasive tool for ALN metastasis assessment, the lack of a standardized ultrasound benchmark dataset has hampered the development and validation of computer-assisted diagnosis (CAD) techniques that could alleviate disparities in access to high-quality screening, particularly in underserved areas. To address these challenges, we present ALN-Ultra, the first large-scale open-access dataset comprising paired ultrasound images and videos from 257 breast cancer patients, along with expert diagnostic and biopsy results. This dataset facilitates the development of machine learning algorithms based on 2D images and 3D videos, aiming to improve diagnostic accuracy and understand the contribution of different modalities of data to ALN metastasis judgment.

## Data source
The original source of video and image data can be download from [[Zenodo]](https://zenodo.org/records/15003119).
We also provide the extracted npy files of all video at [[video]](https://drive.google.com/drive/folders/188LpUn-xj0n8HEgvKjCIvw-RJZPa7SIX?usp=sharing). Each frame in the video modality is resized to 128 px × 200 px to ensure fair comparisons, with 50 frames uniformly sampled from each video.

## Training for video/image
```markdown
