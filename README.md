# GDN
A tensorflow implementation of **Deconvolutional Networks on Graph Data (NeurIPS 2021)**
<p align="center">
</p>
<p align="justify">

## Abstract
In this paper, we consider an inverse problem in graph learning domain – “given the graph representations smoothed by Graph Convolutional Network (GCN), how can we reconstruct the input graph signal?" We propose Graph Deconvolutional Network (GDN) and motivate the design of GDN via a combination of inverse filters in spectral domain and de-noising layers in wavelet domain, as the inverse operation results in a high frequency amplifier and may amplify the noise. We demonstrate the effectiveness of the proposed method on several tasks including graph feature imputation and graph structure generation.
  
This repository provides a Tensorflow implementation of SEAL-CI as described in the paper:
> Deconvolutional Networks on Graph Data. Jia Li, Jiajin Li, Yang Liu, Jianwei Yu, Yueting Li, Hong Cheng. NeurIPS 2021. [[paper]](https://arxiv.org/abs/2110.15528)
  
```
# Dataset
douban

# Model options
```

# Example
```
train from scratch

```
python main.py 
