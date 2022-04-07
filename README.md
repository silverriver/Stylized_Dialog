# Stylized Dialogue Model

Code/data for AAAI'21 paper [Stylized Dialogue Response Generation Using Stylized Unpaired Texts](https://arxiv.org/abs/2009.12719)

We implemented a stylized dialogue model that can capture stylistic text features embedded in stylized unpaired texts. Specifically, an inverse dialogue model is introduced to generate stylized pseudo dialogue pairs, which are further utilized in a joint training process. An effective style routing approach is devised to intensify the stylistic features in the decoder.

## Requirement

The code is tested using python3.5 and pytorch 1.4.0

## How to use

Please see the `TCFC` and `WDJN` folder for more detailed information about the dataset and code usage. Our data and the Chinese pre-trained model can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1l_jLVcpBnGXpLp7yf3lqiw) with extract code of `nmoc`, or downloaded from [Google Drive](https://drive.google.com/drive/folders/1rwWv7gbWQrxDMCOr5fpqVd0jJQF4NQu0?usp=sharing).

## Citation

Please cite our paper if you find this repo useful :)

```
@inproceedings{zheng2021stylized,
  title={Stylized Dialogue Response Generation Using Stylized Unpaired Texts},
  author={Zheng, Yinhe and Chen, Zikai and Zhang, Rongsheng and Huang, Shilei and Mao, Xiaoxi and Huang, Minlie},
  booktitle={AAAI},
  year={2020},
  url={https://arxiv.org/abs/2009.12719}
}
```
