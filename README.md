# Distributed Quantum Model Learning for Traffic Density Estimation

This repository contains the code and resources associated with our paper titled "Distributed Quantum Model Learning for Traffic Density Estimation". Please note that the paper is currently under review for publication.

The code is tested on Ubuntu 22.04 environment (Python3.8, PyTorch1.13.1) with an NVIDIA GeForce RTX 4090.

## Contents

- [Distributed Quantum Model Learning for Traffic Density Estimation]
  - [Introduction](#introduction)
  - [Train](#train)
  - [Test](#test)
  <!-- - [Pretrained Weights](#pretrained-weights) -->
  - [Results](#results)
    - [Quantitative Results](#quantitative-results)
    - [Visual Results](#visual-results)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)

## Introduction
<!-- 
To address the challenges of vehicle counting in real-world applications, we propose the Efficient Vehicular Counting via Privacy-aware Aggregation Network (PANet). PANet integrates a Pyramid Feature Enhancement (PFE) module, which captures multi-scale features effectively and improves the representation of key features. By optimizing channel-wise outputs, the computational complexity is significantly reduced. Moreover, PANet employs a federated learning framework to distribute computational tasks among devices, ensuring robust privacy protection and minimizing the possibility of data leakage. -->

![arch](assets/framework.jpg)
![arch](assets/framework_FL.jpg)

## Train

The training code will be released after the acceptance of this paper
<!-- 1. Prepare the datasets used in the experiment.
2. Modify the data set address in `make_npydata.py` to generate the correct dataset information
3. Modify the dataset, client numbers and other options in `config.py`.
4. After performing the above modifications, you can start the training process by running `python train.py`. -->

## Test

To test PANet, update the `pre` argument in `config.py` with the path to the pretrained model. Then, initiate the testing process by running `python test.py`.

<!-- ## Pretrained Weights -->

<!-- The pretrained weights from [HERE](https://1drv.ms/f/s!Al2dMJC6HUgQrJRUCo3Ighr21TXMwg?e=dSQTCy). -->

## Results

### Quantitative Results

![arch](assets/Carpk_Pucpr.jpg)
![arch](assets/Large_Small.jpg)
![arch](assets/TRANCOS.jpg)
![arch](assets/SHA.jpg)
![arch](assets/Eff.png)

### Visual Results

![arch](assets/sub_carpk.jpg)
![arch](assets/scb_large.jpg)
![arch](assets/sub_trancos.jpg)


## Citation

If you find this code or research helpful, please consider citing our paper:

```BibTeX
@article{Zhai2025distributed,
  title={Distributed Quantum Model Learning for Traffic Density Estimation},
  author={Zhai, Wenzhe, and Cheng, Jing-an and Shabaz, Mohammad and Gao, Mingliang and Zhang, Chen and Wang, Jianyong},
  journal={under_review},
  year={2025},
}
```

## Acknowledgements

This code is built on [DSPI](https://github.com/jinyongch/DSPI) and [FIDTM](https://github.com/dk-liang/FIDTM). We thank the authors for sharing their codes.
