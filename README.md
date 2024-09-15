# LU-M2SR:Lightweight U-Net with Mamba2 for Single Image Super-Resolution
Qingguo Su, Di Zhang, Xinjie Shi, Zezheng Zhao, Kefeng Deng, Kaijun Ren
# About
LU-M2SR is a noval Single Image Super-Resolution method.Single image super-resolution (SISR) technology has made significant progress but also faces the challenge of computational complexity, which limits its application on mobile devices with limited computational resources. In this work, LU-M2SR, a lightweight and effective model based on the Mamba2 with State Space Duality (SSD), is proposed to address this challenge. Specifically, we analyze the key factors influencing the number of parameters in Mamba2, and correspondingly propose a Separable Channel State Space Duality (SC-SSD) module to model long-range dependencies with lower computational complexity. Besides, a Multi-Scale Convolutional Block (MSCB) is designed to effectively capture and aggregate local contextual dependencies across multiple scales. Experiments conducted on five benchmarks demonstrate that the LU-M2SR has equally competitive performance with SOTA methods while holding only 0.563 M Params, and 0.398 G FLOPs.
# Contents

1. [Training](#Training)
1. [Testing](#Testing)
1. [Results](#Results)



## Training
Used training sets can be downloaded as follows:
### Training Datasets
1. [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
2. [Flickr2K](https://www.kaggle.com/datasets/hliang001/flickr2k)

### Train
If you want to run a different configuration, modify `configs\config_setting.py`.
```
python train.py --num_workers=8 --batch_size=64
```
## Testing
Used testing sets can be downloaded as follows:

### Testing Datasets
1. [Set5](https://paperswithcode.com/dataset/set5)
2. [Set14](https://paperswithcode.com/dataset/set14)
3. [BSD100](https://paperswithcode.com/dataset/bsd100)
4. [Urban100](https://paperswithcode.com/dataset/urban100)
5. [Manga109](https://paperswithcode.com/dataset/manga109)
### Test
```
python test.py
```
## Results
<p align="center">
  <img width="900" src="https://github.com/SuMuzi/LU-M2SR/results/results.png">
</p>
