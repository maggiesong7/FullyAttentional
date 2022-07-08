# Fully Attentional Network for Semantic Segmentation

PyTorch code for FLANet (AAAI 2022).

Fully Attentional Network for Semantic Segmentation  
Qi Song, Jie Li, Chenghong Li and Hao Guo   
AAAI, 2022  
[[Paper](https://arxiv.org/pdf/2112.04108.pdf)]

Abstract: Recent non-local self-attention methods have proven to be effective in capturing long-range dependencies for semantic segmentation. These methods usually form a similarity map of \(\mathbb{R}^{C\times C}\) (by compressing spatial dimensions) or \(\mathbb{R}^{HW\times HW}\) (by compressing channels) to describe the feature relations along either channel or spatial dimensions, where \(C\) is the number of channels, \(H\) and \(W\) are the spatial dimensions of the input feature map. However, such practices tend to condense feature dependencies along the other dimensions, hence causing attention missing, which might lead to inferior results for small/thin categories or inconsistent segmentation inside large objects. To address this problem, we propose a new approach, namely Fully Attentional Network (FLANet), to encode both spatial and channel attentions in a single similarity map while maintaining high computational efficiency. Specifically, for each channel map, our FLANet can harvest feature responses from all other channel maps, and the associated spatial positions as well, through a novel fully attentional module. 

## Requirements:
```
pytorch==1.6.0
```

## Citation
If you found our method useful in your research, please consider citing

```
@inproceedings{song2022fully,
  title={Fully attentional network for semantic segmentation},
  author={Song, Qi and Li, Jie and Li, Chenghong and Guo, Hao and Huang, Rui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={2},
  pages={2280--2288},
  year={2022}
}
```
