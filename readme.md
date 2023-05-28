## Lightweight Stepless Super-Resolution of Remote Sensing Images via Saliency-Aware Dynamic Routing Strategy

by Hanlin Wu, Ning Ni, and Libao Zhang, details are in [paper](https://arxiv.org/abs/2210.07598).

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

1. Download the training datset from this [url](https://github.com/hanlinwu/SalDRN/releases/download/v1.0.0/Train_and_Test_data.zip). 
2. Unzip the downloaded dataset, and put the files on path: `load/SalCSSR-339`
3. Change the `hr_path` and `sal_path` in `config/your_config_file.yaml`
4. Do training:
   ```
   python train.py --config config/your_config_file.yaml
   ```

### Test:
1. Unzip the benchmark dataset, and put the files on path: `load/benchmark`
```
python test.py --checkpoint logs/your_checkpoint_path
```
