# Stylized Dialogue Model

Code/data for AAAI'21 paper [Stylized Dialogue Response Generation Using Stylized Unpaired Texts](https://arxiv.org/abs/2009.12719)

We implemented a stylized dialogue model that can capture stylistic text features embedded in stylized unpaired texts. Specifically, an inverse dialogue model is introduced to generate stylized pseudo dialogue pairs, which are further utilized in a joint training process. An effective style routing approach is devised to intensify the stylistic features in the decoder.

## Requirement

The code is tested using python3.5 and pytorch 1.4.0

## How to use

Please see the `TCFC` and `WDJN` folder for more detailed information about the dataset and code usage.

## Citation

Please cite our paper if your find this repo useful :)

```
@inproceedings{zheng2021stylized,
  title={Stylized Dialogue Response Generation Using Stylized Unpaired Texts},
  author={Zheng, Yinhe and Chen, Zikai and Zhang, Rongsheng and Huang, Shilei and Mao, Xiaoxi and Huang, Minlie},
  booktitle={AAAI},
  year={2020},
  url={https://arxiv.org/abs/2009.12719}
}
```
