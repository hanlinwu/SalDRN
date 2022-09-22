## Lightweight Stepless Super-Resolution of Remote Sensing Images via Saliency-Aware Dynamic Routing Strategy

by Hanlin Wu, Ning Ni, and Libao Zhang, details are in paper.

## Usage

### Clone the repository:
```
git clone https://github.com/hanlinwu/SalDRN.git
```
## Requirements:
- pytorch==1.10.0
- pytorch-lightning==1.5.5
- numpy
- opencv-python
- easydict
- tqdm

### Train:

1. Unzip the the downloaded dataset, and put the files on path: `load/dataset_name`
2. Change the `hr_path` and `sal_path` in `config/your_config_file.yaml`
3. Do training:
   ```
   python train.py --config config/your_config_file.yaml
   ```

### Test:
```
python test.py --checkpoint your_checkpoint_path
```