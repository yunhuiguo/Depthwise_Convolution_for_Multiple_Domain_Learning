# Depthwise Convolution is All You Need for Learning Multiple Visual Domains
### https://www.aaai.org/ojs/index.php/AAAI/article/view/4851
Authors: Yunhui Guo*, Yandong Li*, Rogerio Feris, Liqiang Wang, Tajana Rosing     
*Equal contribution

## Abstract
There is a growing interest in designing models that can deal with images from different visual domains. If there exists a universal structure in different visual domains that can be captured via a common parameterization, then we can use a single model for all domains rather than one model per domain. A model aware of the relationships between different domains can also be trained to work on new domains with less resources. However, to identify the reusable structure in a model is not easy. In this paper, we propose a multi-domain learning architecture based on depthwise separable convolution. The proposed approach is based on the assumption that images from different domains share cross-channel correlations but have domain-specific spatial correlations. The proposed model is compact and has minimal overhead when being applied to new domains. Additionally, we introduce a gating mechanism to promote soft sharing between different domains. We evaluate our approach on Visual Decathlon Challenge, a benchmark for testing the ability of multi-domain models. The experiments show that our approach can achieve the highest score while only requiring 50% of the parameters compared with the state-of-the-art approaches.

## Training

Please follow https://www.robots.ox.ac.uk/~vgg/decathlon/ to download the datasets,

Example runs are:
```bash
$ python main.py

$ python submit_test.py

```

If you find this repository useful in your own research, please consider citing:
```
@inproceedings{guo2019depthwise,
  title={Depthwise convolution is all you need for learning multiple visual domains},
  author={Guo, Yunhui and Li, Yandong and Wang, Liqiang and Rosing, Tajana},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={8368--8375},
  year={2019}
}
```
