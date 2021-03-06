# DL-BET
Deep Learning based Brain Extraction Tool 

## Preprocessing of Small Animal MRI Scans
1. NIFTI conversion from BRUKER raw data  
2. Isotropic upsampling
3. N4 Bias field correction  

## Brain Mask Generation
### Installation
1. Create and activate a conda environment.
2. Install PyTorch. Visit https://pytorch.org/get-started/locally/ for installation details.
3. Install other required packages.
4. Clone this repository.

### Usage
1. Save training, validation, and test date to a directory called 'dataset'. A randomized 80-10-10 split is suggested. Define folder paths in configs_settings.ini.
2. Run 'python3 DL-BET.py' to begin training model.
3. Run 'python3 DL-BET_test.py' after training has completed to generate test outputs, saved to 'results'.
