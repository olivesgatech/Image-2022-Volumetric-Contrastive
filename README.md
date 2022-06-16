# Volumetric Supervised Contrastive Learning for Seismic Semantic Segmentation
## Dataset

## Code Usage
1. Initiate contrastive learning pre-training with main_supcon.py. Identify the labels to train with by choosing the appropriate discretization level csv file.
2. Do semantic segmentation training within main_seismic_semantic.py by identifying the checkpoint file generated in step 1 and choosing the test fold csv file of interest. 
## Links

**Associated Website**: https://ghassanalregib.info/

**Code Acknowledgement**: Code is based off of https://github.com/HobbitLong/SupContrast.git.

## Citations

If you find the work useful, please include the following citation in your work:

>@inproceedings{kokilepersaud2022volumetric,\
  title={Volumetric Supervised Contrastive Learning for Seismic Semantic Segmentation},\
  author={Kokilepersaud, Kiran, and Prabhushankar, Mohit and AlRegib, Ghassan},\
  booktitle={The International Meeting for
Applied Geoscience & Energy},\
  year={2022}\
}
