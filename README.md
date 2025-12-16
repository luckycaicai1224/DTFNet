# DTFNet: A Dual-Modal Time-Frequency Fusion Network for Non-Stationary Time Series Modeling.

## Getting Started

1. Install requirements. 

```
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu118

```

2. Download data. You can download all the datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Create a seperate folder ```./dataset``` and put all the csv files in the directory.

### Dual-Modal Time-Frequency Fusion Network (DTFNet) 

3. Training. All the scripts are in the directory ```./scripts/DTFNet```. For example,

```
sh ./scripts/DTFNet/long_term_forecast/etth1.sh
```


## Acknowledgement

We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/linxi20/CPAT

https://github.com/TROUBADOUR000/TimeFilter

https://github.com/huangst21/TimeKAN

https://github.com/SDUYanDong/TFP-Mixer

https://github.com/Hank0626/PDF

https://github.com/yuqinie98/PatchTST

https://github.com/cure-lab/LTSF-Linear


