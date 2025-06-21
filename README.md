# MMSBR
This is our implementation for the paper:

_Beyond Co-occurrence: Multi-modal Session-based Recommendation_ 

Xiaokun Zhang, Bo Xu, Fenglong Ma, Chenliang Li, Liang Yang, Hongfei Lin

_at TKDE, 2023_

Note that, as far as we know, this is the first efforts to consider multi-modal information (images, text and price) in session-based recommendation. Several kinds of information are included in the work for user behavior modeling. Therefore, the data preprocessing flow maybe a little complex. I have tried my best to detail the work flow of processing data. Please see the comments in the codes for code understand.

Download path of datasets:

  Amazon: [https://www.kaggle.com/mkechinov/ecommerce-events-history-in-cosmeticsshop](http://jmcauley.ucsd.edu/data/amazon/)

We also uploaded all three preprocessed datasets under the folder 'datasets'. You can directly use them to reproduce our results. You can also process your own datasets via the preprocess code we provide in the file 'preprocess'. Note that, please read the comments in the .py files to learn the purpose of the files. 

Please cite our paper if you use our codes. Thanks!
```
@article{MMSBR,
  author={Zhang, Xiaokun and Xu, Bo and Ma, Fenglong and Li, Chenliang and Yang, Liang and Lin, Hongfei},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Beyond Co-Occurrence: Multi-Modal Session-Based Recommendation}, 
  year={2024},
  volume={36},
  number={4},
  pages={1450-1462},
  keywords={Probabilistic logic;Behavioral sciences;Numerical models;Transformers;Semantics;Graph neural networks;Fuses;Hierarchical pivot transformer;multi-modal learning;probabilistic modeling;pseudo-modality contrastive learning;session-based recommendation},
  doi={10.1109/TKDE.2023.3309995}}

```

In case that you have any difficulty about the implementation or you are interested in our work,  please feel free to communicate with me by:

Author: Xiaokun Zhang (dawnkun1993@gmail.com)

Also, welcome to visit my academic homepage:

https://zhang-xiaokun.github.io/

