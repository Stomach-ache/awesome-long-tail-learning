# Awesome Long-Tailed Learning [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

This repo pays specially attention to the long-tailed distribution, where labels follow a long-tailed or power-law distribution in the training dataset or/and test dataset. Related papers are sumarized, including its application in computer vision, in particular image classification, and extreme multi-label learning (XML), in particular text categorization.

### :high_brightness: Updated 2022-10-14


- Long-tailed Distribution
  * [Long-tailed Learning](#Long-tailed-Learning)
  * [Long-Tailed Semi-Supervised Learning](#Long-Tailed-Semi-Supervised-Learning-Papers)
  * [Long-Tailed Learning with Noisy Labels](#Long-Tailed-Learning-with-Noisy-Labels-Papers)
  * [Long-Tailed Federated Learning](#Long-Tailed-Federated-Learning-Papers)
  * [eXtreme Multi-label Learning for Information Retrieval](#eXtreme-Multi-label-Learning-for-Information-Retrieval)
    + [Binary Relevance](#Binary-Relevance)
    + [Tree-based Methods](#Tree-based-Methods)
    + [Embedding-based Methods](#Embedding-based-Methods)
    + [Speed-up and Compression](#Speed-up-and-Compression)
    + [Noval XML Setups](#Noval-XML-Settings)
    + [Theoritical Studies](#Theoritical-Studies)
    + [Text Classification](#Text-Classification)
    + [Others](#Others)
  


<!-- toc -->


# Long-tailed Learning

### Type of Long-Tailed Learning Methods

| Type        | `TST`          | `IS`           | `CBS`                   | `CLW`                 | `NC`                  | `ENS`              | `DA`     |
|:----------- |:-------------:|:--------------:|:----------------------: |:---------------------:|:----------------------:|:-----------------:|:-----------:|
| Meaning | Two-Stage Training | Instance Sampling | Class-Balanced Sampling | Class-Level Weighting | Normalized Classifier | Ensemble | Data Augmentation |

### Long-Tailed Learning Workshops


| Year       | Venue     | Title  | Remark
| ------------- |:-------------:| --------------:|------------:|
|2021 | CVPR | [Open World Vision](http://www.cs.cmu.edu/~shuk/open-world-vision.html) | long-tail, open-set, streaming labels|
|2021 | CVPR | [Learning from Limited and Imperfect Data (L2ID)](https://l2id.github.io/) | label noise, SSL, long-tail |


### Long-Tailed Regression Papers
| Year       | Venue     | Title  | Remark
| ------------- |:-------------:| --------------:|------------:|
|2022 | CVPR | [Balanced MSE for Imbalanced Visual Regression](https://arxiv.org/pdf/2203.16427.pdf) | |
|2021 | OpenReview | [LIFTING IMBALANCED REGRESSION WITH SELF- SUPERVISED LEARNING](https://openreview.net/pdf?id=8Dhw-NmmwT3) | iclr rejected |
|2021 | ICML | [Delving into Deep Imbalanced Regression](https://arxiv.org/pdf/2102.09554.pdf) | [code](https://github.com/YyzHarry/imbalanced-regression) |


### Long-Tailed Classification Papers

| Year       | Venue     | Title  | Remark
| ------------- |:-------------:| --------------:|------------:|
|2022 | NeurIPS | [Self-Supervised Aggregation of Diverse Experts for Test-Agnostic Long-Tailed Recognition](https://arxiv.org/pdf/2107.09249.pdf) | [code](https://github.com/Vanint/SADE-AgnosticLT)|
|2022 | arXiv | [Learning to Re-weight Examples with Optimal Transport for Imbalanced Classification](https://arxiv.org/pdf/2208.02951.pdf) | |
|2022 | TPAMI | [Key Point Sensitive Loss for Long-tailed Visual Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9848833) | |
|2022 | IJCV | [A Survey on Long-Tailed Visual Recognition](https://link.springer.com/content/pdf/10.1007/s11263-022-01622-8.pdf) | survey|
| 2022 | Arxiv | [Neural Collapse Inspired Attraction-Repulsion-Balanced Loss for Imbalanced Learning](https://arxiv.org/pdf/2204.08735.pdf) |
|2022 | ICLR | [OPTIMAL TRANSPORT FOR LONG-TAILED RECOGNI- TION WITH LEARNABLE COST MATRIX](https://openreview.net/pdf?id=t98k9ePQQpn) | |
|2022 | ICLR | [SELF-SUPERVISED LEARNING IS MORE ROBUST TO DATASET IMBALANCE](https://openreview.net/pdf?id=4AZz9osqrar) | |
|2022 | AAAI | [Cross-Domain Empirical Risk Minimization for Unbiased Long-tailed Classification](https://arxiv.org/pdf/2112.14380.pdf) | [code](https://github.com/BeierZhu/xERM) |
|2021 | NeurIPS | [Improving Contrastive Learning on Imbalanced Seed Data via Open-World Sampling](https://papers.nips.cc/paper/2021/hash/2f37d10131f2a483a8dd005b3d14b0d9-Abstract.html) | |
|2021 | NeurIPS | [Towards Calibrated Model for Long-Tailed Visual Recognition from Prior Perspective](https://papers.nips.cc/paper/2021/hash/39ae2ed11b14a4ccb41d35e9d1ba5d11-Abstract.html) | [code](https://github.com/XuZhengzhuo/Prior-LT), mixup+LA |
|2021 |Arxiv | [HAR: Hardness Aware Reweighting for Imbalanced Datasets](https://poloclub.github.io/papers/21-bigdata-har.pdf)|  |
|2021 |Arxiv | [Feature Generation for Long-tail Classification](https://arxiv.org/pdf/2111.05956.pdf) |
|2021 |Arxiv | [Label-Aware Distribution Calibration for Long-tailed Classification](https://arxiv.org/pdf/2111.04901.pdf)  |
|2021 | Arxiv | [Self-supervised Learning is More Robust to Dataset Imbalance](https://arxiv.org/pdf/2110.05025.pdf) | |
|2021 | Arixiv | [Long-tailed Distribution Adaptation](https://arxiv.org/pdf/2110.02686.pdf) | |
|2021 | Arxiv | [LEARNING FROM LONG-TAILED DATA WITH NOISY LABELS](https://arxiv.org/pdf/2108.11096.pdf) | |
|2021 | ICCV  | [Self Supervision to Distillation for Long-Tailed Visual Recognition](https://arxiv.org/abs/2109.04075) | |
|2021 | ICCV | [Distilling Virtual Examples for Long-tailed Recognition](https://cs.nju.edu.cn/wujx/paper/ICCV2021_DiVE.pdf) |
|2021 | CVPR  | [Contrastive Learning based Hybrid Networks for Long-Tailed Image Classification](https://arxiv.org/pdf/2103.14267.pdf) | |
|2021 | CVPR  | [MetaSAug: Meta Semantic Augmentation for Long-Tailed Visual Recognition](https://arxiv.org/pdf/2103.12579.pdf) | |
|2021 | CVPR  | [Disentangling Label Distribution for Long-tailed Visual Recognition](https://arxiv.org/pdf/2012.00321.pdf) | |
|2021 | CVPR  | [Long-Tailed Multi-Label Visual Recognition by Collaborative Training on Uniform and Re-Balanced Samplings](https://openaccess.thecvf.com/content/CVPR2021/papers/Guo_Long-Tailed_Multi-Label_Visual_Recognition_by_Collaborative_Training_on_Uniform_and_CVPR_2021_paper.pdf) | |
|2021 | CVPR  | [Seesaw Loss for Long-Tailed Instance Segmentation](https://arxiv.org/pdf/2008.10032.pdf) | |
|2021 | ICLR  | [Exploring balanced feature spaces for representation learning](https://openreview.net/pdf?id=OqtLIabPTit) | |
|2021 | ICLR  | [IS LABEL SMOOTHING TRULY INCOMPATIBLE WITH KNOWLEDGE DISTILLATION: AN EMPIRICAL STUDY](https://openreview.net/pdf?id=PObuuGVrGaZ#:~:text=Our%20answer%20is%20No.,the%20predictive%20performance%20of%20students.) | |
|2021 | Arxiv | [Improving Long-Tailed Classification from Instance Level](https://arxiv.org/pdf/2104.06094.pdf) | |
|2021 | Arxiv | [ResLT: Residual Learning for Long-tailed Recognition](https://arxiv.org/pdf/2101.10633v2.pdf) |
|2021 | Arxiv | [Improving Long-Tailed Classification from Instance Level](https://arxiv.org/pdf/2104.06094v1.pdf) |
|2021 |Arxiv | [Disentangling Sampling and Labeling Bias for Learning in Large-Output Spaces](https://arxiv.org/pdf/2105.05736.pdf) | by Google |
|2021 | Arxiv | [Breadcrumbs: Adversarial Class-Balanced Sampling for Long-tailed Recognition](https://arxiv.org/pdf/2105.00127.pdf) | |
|2021 | Arxiv | [Procrustean Training for Imbalanced Deep Learning](https://arxiv.org/pdf/2104.01769.pdf) | |
|2021 | Arxiv| [Balanced Knowledge Distillation for Long-tailed Learning](https://arxiv.org/pdf/2104.10510.pdf) | `CBS`+`IS`, [Code](https://github.com/EricZsy/BalancedKnowledgeDistillation)|
|2021 | Arxiv| [Class-Balanced Distillation for Long-Tailed Visual Recognition](https://arxiv.org/pdf/2104.05279.pdf) | `ENS`+`DA`+`IS`, by Google Research |
|2021 | Arxiv| [Distributional Robustness Loss for Long-tail Learning](https://arxiv.org/pdf/2104.03066.pdf) | `TST`+`CBS` |
|2021 | CVPR| [Improving Calibration for Long-Tailed Recognition](https://arxiv.org/pdf/2104.00466.pdf) | `DA`+`TST`, [Code](https://github.com/Jia-Research-Lab/MiSLAS) |
|2021 | CVPR| [Distribution Alignment: A Unified Framework for Long-tail Visual Recognition](https://arxiv.org/pdf/2103.16370.pdf) | `TST` |
|2021 | CVPR | [Adversarial Robustness under Long-Tailed Distribution](https://arxiv.org/pdf/2104.02703.pdf) | |
|2021 | ICLR | [HETEROSKEDASTIC AND IMBALANCED DEEP LEARNING WITH ADAPTIVE REGULARIZATION](https://openreview.net/pdf?id=mEdwVCRJuX4) | [Code](https://github.com/kaidic/HAR) |
|2021 | ICLR | [LONG-TAILED RECOGNITION BY ROUTING DIVERSE DISTRIBUTION-AWARE EXPERTS](https://arxiv.org/pdf/2010.01809.pdf) | `ENS`+`NC`, [Code](https://github.com/frank-xwang/RIDE-LongTailRecognition), by Zi-Wei Liu |
|2021 | ICLR | [Long-Tail Learning via Logit Adjustment](https://arxiv.org/pdf/2007.07314.pdf) | by Google |
|2021 | AAAI | [Bag of Tricks for Long-Tailed Visual Recognition with Deep Convolutional Neural Networks](https://cs.nju.edu.cn/wujx/paper/AAAI2021_Tricks.pdf) | |
|2021           | Arxiv | [Learning From Multiple Experts: Self-paced Knowledge Distillation for Long-tailed Classification](https://arxiv.org/pdf/2001.01536.pdf) | |
|2020 | Arxiv| [ELF: An Early-Exiting Framework for Long-Tailed Classification](https://arxiv.org/pdf/2006.11979.pdf) | |
|2020 | CVPR | [Rethinking Class-Balanced Methods for Long-Tailed Visual Recognition from a Domain Adaptation Perspective](https://arxiv.org/pdf/2003.10780.pdf) | |
|2020 | CVPR | [Equalization Loss for Long-Tailed Object Recognition](https://arxiv.org/pdf/2003.05176.pdf) | |
|2020 | CVPR | [Deep Representation Learning on Long-tailed Data: A Learnable Embedding Augmentation Perspective](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Deep_Representation_Learning_on_Long-Tailed_Data_A_Learnable_Embedding_Augmentation_CVPR_2020_paper.pdf)
|2020 | ICLR | [Decoupling representation and classifier for long-tailed recognition](https://openreview.net/pdf?id=r1gRTCVFvB) | [Code](https://github.com/facebookresearch/classifier-balancing) |
|2020 | NeurIPS | [Balanced Meta-Softmax for Long-Tailed Visual Recognition](https://arxiv.org/pdf/2007.10740.pdf) | |
|2020 | NeurIPS | [Rethinking the Value of Labels for Improving Class-Imbalanced Learning](https://arxiv.org/pdf/2006.07529.pdf) | [Code](https://github.com/YyzHarry/imbalanced-semi-self) |
|2020 | CVPR | [Bbn: Bilateral-branch network with cumulative learning for long-tailed visual recognition](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_BBN_Bilateral-Branch_Network_With_Cumulative_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2020_paper.pdf) | [Code](https://github.com/Megvii-Nanjing/BBN) |
|2019 | NeurIPS | [Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss](https://arxiv.org/pdf/1906.07413.pdf) | [Code](https://github.com/kaidic/LDAM-DRW) |
|2019           | CVPR  | [Large-Scale Long-Tailed Recognition in an Open World](https://arxiv.org/pdf/1904.05160.pdf) | [Code](https://github.com/zhmiao/OpenLongTailRecognition-OLTR), [bibtex](https://dblp.uni-trier.de/rec/bibtex/conf/cvpr/0002MZWGY19), by CUHK |
|2018 | - | [iNatrualist. The inaturalist 2018 competition dataset](https://github.com/visipedia/inat_comp/tree/master/2018) | long-tailed dataset |
|2017 | Arxiv | [The Devil is in the Tails: Fine-grained Classification in the Wild](https://arxiv.org/pdf/1709.01450.pdf) | |
|2017 | NeurIPS | Learning to model the tail | |

----


### Long-Tailed Semi-Supervised Learning Papers

| Year       | Venue     | Title  | Remark
| ------------- |:-------------:| --------------:|------------:|
|2022 | CVPR | [DASO: Distribution-Aware Semantics-Oriented Pseudo-label for Imbalanced Semi-Supervised Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Oh_DASO_Distribution-Aware_Semantics-Oriented_Pseudo-Label_for_Imbalanced_Semi-Supervised_Learning_CVPR_2022_paper.pdf) | [code](https://github.com/ytaek-oh/daso) |
|2022 | MLJ | [Transfer and Share: Semi-Supervised Learning from Long-Tailed Data](https://arxiv.org/abs/2205.13358) | [code](https://github.com/Stomach-ache/TRAS) |
|2022 | ICML | [Smoothed Adaptive Weighting for Imbalanced Semi-Supervised Learning: Improve Reliability Against Unknown Distribution Data](https://proceedings.mlr.press/v162/lai22b.html) | [code](https://github.com/ZJUJeffLai/SAW_SSL) |
|2022 | ICLR | [THE RICH GET RICHER: DISPARATE IMPACT OF SEMI-SUPERVISED LEARNING](https://openreview.net/pdf?id=DXPftn5kjQK) | |
|2022 | ICLR | [ON NON-RANDOM MISSING LABELS IN SEMI-SUPERVISED LEARNING](https://openreview.net/pdf?id=6yVvwR9H9Oj) | |
|2022 | OpenReview | [UNIFYING DISTRIBUTION ALIGNMENT AS A LOSS FOR IMBALANCED SEMI-SUPERVISED LEARNING](https://openreview.net/forum?id=HHUSDJb_4KJ)
|2021 |NeurIPS | [ABC: Auxiliary Balanced Classifier for Class-imbalanced Semi-supervised Learning](https://arxiv.org/abs/2110.10368) |
|2021 | Arxiv | [CoSSL: Co-Learning of Representation and Classifier for Imbalanced Semi-Supervised Learning](https://arxiv.org/pdf/2112.04564.pdf)
|2021 | CVPR | [CReST: A Class-Rebalancing Self-Training Framework for Imbalanced Semi-Supervised Learning](https://arxiv.org/pdf/2102.09559.pdf) | by Google, [Code](https://github.com/google-research/crest), Tensorflow |
|2021 | Arxiv | [DISTRIBUTION-AWARE SEMANTICS-ORIENTED PSEUDO-LABEL FOR IMBALANCED SEMI-SUPERVISED LEARNING](https://arxiv.org/pdf/2106.05682.pdf) | SSL, [Code](https://github.com/ytaek-oh/daso) |
|2020 | NeurIPS | [Distribution Aligning Refinery of Pseudo-label for Imbalanced Semi-supervised Learning](https://papers.nips.cc/paper/2020/file/a7968b4339a1b85b7dbdb362dc44f9c4-Paper.pdf) | [Code](https://github.com/bbuing9/DARP) |

----


### Long-Tailed Learning with Noisy Labels Papers

| Year       | Venue     | Title  | Remark
| ------------- |:-------------:| --------------:|------------:|
|2022 | ECCV | [Identifying Hard Noise in Long-Tailed Sample Distribution](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860725.pdf) | [code](https://github.com/yxymessi/H2E-Framework), large datasets|
|2022 | ICLR | [SAMPLE SELECTION WITH UNCERTAINTY OF LOSSES FOR LEARNING WITH NOISY LABELS](https://openreview.net/pdf?id=xENf4QUL4LW) | |
|2022 |PAKDD | [Prototypical Classifier for Robust Class-Imbalanced Learning](https://arxiv.org/pdf/2110.11553.pdf) | [code](https://github.com/Stomach-ache/PCL) |
|2021 |Arxiv | [ROBUST LONG-TAILED LEARNING UNDER LABEL NOISE](https://arxiv.org/pdf/2108.11569.pdf) | [code](https://github.com/Stomach-ache/RoLT)|

----

### Long-Tailed Federated Learning Papers
| Year       | Venue     | Title  | Remark
| ------------- |:-------------:| --------------:|------------:|
|2022 |IJCAI | [Federated Learning on Heterogeneous and Long-Tailed Data via Classifier Re-Training with Federated Features](https://arxiv.org/pdf/2204.13399.pdf) |


# eXtreme Multi-label Learning for Information Retrieval


## Binary Relevance

| Year       | Venue       | Title  | Remark
| ------------- |:-------------:| --------------:|------------:|
|2019 | Machine learning | [Data Scarcity, Robustness and Extreme Multi-label Classification](https://link.springer.com/article/10.1007/s10994-019-05791-5) | |
|2019 | WSDM | [Slice: Scalable linear extreme classifiers trained on 100 million labels for related searches](http://manikvarma.org/pubs/jain19.pdf) |
|2017 | KDD | PPDSparse: A Parallel Primal-Dual Sparse Method for Extreme Classification |
|2017 | AISTATS | Label Filters for Large Scale Multilabel Classification |
|2016 | WSDM | DiSMEC - Distributed Sparse Machines for Extreme Multi-label Classification |
|2016 | ICML | [PD-Sparse: A Primal and Dual Sparse Approach to Extreme Multiclass and Multilabel Classification](http://proceedings.mlr.press/v48/yenb16.pdf) |

----

## Tree-based Methods

| Year       | Venue       | Title  | Remark
| ------------- |:-------------:| --------------:|------------:|
|2021 | KDD | [Extreme Multi-label Learning for Semantic Matching in Product Search](https://arxiv.org/pdf/2106.12657.pdf) | by Amazon, [code](https://github.com/amzn/pecos) |
|2020 | arXiv | [Probabilistic Label Trees for Extreme Multi-label Classification](https://arxiv.org/pdf/2009.11218.pdf) | PLT survey, [code](https://github.com/busarobi/XMLC) |
|2020 | arXiv | [Online probabilistic label trees](https://arxiv.org/pdf/2007.04451.pdf) | |
|2020           | AISTATS  | [LdSM: Logarithm-depth Streaming Multi-label Decision Trees](https://arxiv.org/pdf/1905.10428.pdf) | Instance tree,[c++ code](https://github.com/mmajzoubi/LdSM) |
|2019 | NeurIPS | [AttentionXML: Extreme Multi-Label Text Classification with Multi-Label Attention Based Recurrent Neural Networks](https://arxiv.org/pdf/1811.01727.pdf) | Label tree |
|2019   | arXiv | [Bonsai - Diverse and Shallow Trees for Extreme Multi-label Classification](https://arxiv.org/pdf/1904.08249.pdf) | Label tree |
|2018 | ICML | [CRAFTML, an Efficient Clustering-based Random Forest for Extreme Multi-label Learning](http://proceedings.mlr.press/v80/siblini18a/siblini18a.pdf) | Instance tree |
|2018           | WWW | [Parabel: Partitioned Label Trees for Extreme Classification with Application to Dynamic Search Advertising](http://manikvarma.org/pubs/prabhu18b.pdf) | Label tree...by Manik Varma |
|2016 | ICML | [Extreme F-Measure Maximization using Sparse Probability Estimates](https://pdfs.semanticscholar.org/b148/75d1e1850121d8720c39f853af5f455ecc44.pdf) | Label tree |
|2016 | KDD | [Extreme Multi-label Loss Functions for Recommendation, Tagging, Ranking & Other Missing Label Applications](http://manikvarma.org/pubs/jain16.pdf) | Instance tree |
|2014 | KDD | [A Fast, Accurate and Stable Tree-classifier for eXtreme Multi-label Learning](http://manikvarma.org/pubs/prabhu14.pdf) | Instance tree, [python implementation](https://github.com/Refefer/fastxml) |
|2013 | ICML | [Label Partitioning For Sublinear Ranking](http://www.thespermwhale.com/jaseweston/papers/label_partitioner.pdf) | Label tree |
|2013 | WWW | [Multi-Label Learning with Millions of Labels: Recommending Advertiser Bid Phrases for Web Pages](http://manikvarma.org/pubs/agrawal13.pdf) | Instance tree, Random Forest, Gini Index |
|2011 | NeurIPS | [Efficient label tree learning for large scale object recognition](http://vision.stanford.edu/pdf/NIPS2011_0391.pdf) | Label tree, multi-class |
|2010 | NeurIPS | [Label embedding trees for large multi-class tasks](https://papers.nips.cc/paper/4027-label-embedding-trees-for-large-multi-class-tasks.pdf) | Label tree, multi-class |
|2008 | ECML Workshop | [Effective and Efficient Multilabel Classification in Domains with Large Number of Labels](http://lpis.csd.auth.gr/publications/tsoumakas-mmd08.pdf) | Label tree |


----
## Embedding-based Methods

| Year       | Venue      | Title  | Remark
| ------------- |:-------------:| --------------:|------------:|
|2019 | AAAI | [Distributional Semantics Meets Multi-Label Learning](https://aaai.org/ojs/index.php/AAAI/article/view/4260) | [bibtex](https://dblp.uni-trier.de/rec/bibtex/conf/aaai/0001WNK0R19)|
|2019 | arXiv | [Ranking-Based Autoencoder for Extreme Multi-label Classification](https://arxiv.org/pdf/1904.05937.pdf) | |
|2019 | NeurIPS | [Breaking the Glass Ceiling for Embedding-Based Classifiers for Large Ouput Spaces](https://ai.google/research/pubs/pub48660/) | by Google Research|
|2017 | KDD | [AnnexML: Approximate Nearest Neighbor Search for Extreme Multi-label Classification](https://dl.acm.org/doi/pdf/10.1145/3097983.3097987?casa_token=RuuvNarln6EAAAAA:CGN8q-mIboNoQrXybH-pFWF9VboN0cVjdJcTfoYDCEju-ZGriu5MqTDY2jJ-_DOc3fMO4nQftYAIsg) | |
|2015 | NeurIPS | [Sparse Local Embeddings for Extreme Multi-label Classification](http://manikvarma.org/pubs/bhatia15.pdf) | |
|2014 | ICML | Large-scale Multi-label Learning with Missing Labels |
|2014 | ICML | Multi-label Classification via Feature-aware Implicit Label Space Encoding |
|2013 | ICML | Efficient Multi-label Classification with Many Labels |
|2012 | NeurIIPS | Feature-aware Label Space Dimension Reduction for Multi-label Classification |
|2011 | IJCAI | [WSABIE: Scaling Up To Large Vocabulary Image Annotation](file:///C:/Users/Admin/Downloads/2926-16078-1-PB.pdf) | [bibtex](https://dblp.uni-trier.de/rec/bibtex/conf/ijcai/WestonBU11) |
|2009 | NeurIPS | Multi-Label Prediction via Compressed Sensing |
|2008 | KDD | Extracting Shared Subspaces for Multi-label Classification |

----

## Speed-up and Compression

| Year       | Venue       | Title  | Remark
| ------------- |:-------------:| --------------:|------------:|
|2020 | KDD | [Large-Scale Training System for 100-Million Classification at Alibaba](https://dl.acm.org/doi/pdf/10.1145/3394486.3403342?casa_token=ZGu6l5eQUeYAAAAA:ehAedQ8l6p0WRVX7l81E6lRYQPlqnTwx9iEJzXqO46gE5q23JxamlNMwcFKPjFoC38-EsqcVl-kC2w) | Applied Data Science Track|
|2020 |arXiv | [SOLAR: Sparse Orthogonal Learned and Random Embeddings](https://arxiv.org/pdf/2008.13225.pdf) | |
|2020 | ICLR | [EXTREME CLASSIFICATION VIA ADVERSARIAL SOFTMAX APPROXIMATION](https://arxiv.org/pdf/2002.06298.pdf) | |
|2019 | AISTATS | [Stochastic Negative Mining for Learning with Large Output Spaces](http://www.sanjivk.com/SNM_AISTATS19.pdf) | by Google|
|2019 | NeurIPS | [Extreme Classification in Log Memory using Count-Min Sketch: A Case Study of Amazon Search with 50M Products](https://openreview.net/pdf?id=BkgViHSxLH) | Rice University, [bibtex](https://dblp.uni-trier.de/rec/bibtex/journals/corr/abs-1910-13830) |
|2019     | arXiv | [An Embarrassingly Simple Baseline for eXtreme Multi-label Prediction](https://arxiv.org/pdf/1912.08140.pdf) | |
|2019			|    arXiv   |    [Accelerating Extreme Classification via Adaptive Feature Agglomeration](https://arxiv.org/pdf/1905.11769.pdf)  |   [bibtex](https://dblp.uni-trier.de/rec/bibtex/conf/ijcai/JalanK19), authors from IIT    |
|2019     | SDM     | [Fast Training for Large-Scale One-versus-All Linear Classifiers using Tree-Structured Initialization](https://www.cs.ubc.ca/~mpf/pdfs/2019-one-vs-all.pdf) | [code](https://github.com/fanghgit/XMC) [bibtex](https://dblp.uni-trier.de/rec/bibtex/conf/sdm/FangCHF19) |

----

## Noval XML Settings
| Year       | Venue     | Title  | Remark
| ------------- |:-------------:| --------------:|------------:|
|2020 | arXiv | [Extreme Multi-label Classification from Aggregated Labels](https://arxiv.org/pdf/2004.00198.pdf) | by Inderjit Dhillon. This paper considers multi-instance learning in XML |
| 2020 | arXiv | [Unbiased Loss Functions for Extreme Classification With Missing Labels](https://arxiv.org/pdf/2007.00237.pdf) | by Rohit Babbar. Missing labels|
| 2020 | ICML | [Deep Streaming Label Learning](https://proceedings.icml.cc/static/paper_files/icml/2020/230-Paper.pdf) | [code](https://github.com/DSLLcode/DSLL), by Dacheng Tao, streaming multi-label learning |
| 2016 | arXiv | [Streaming Label Learning for Modeling Labels on the Fly](https://arxiv.org/pdf/1604.05449.pdf) | by Dacheng Tao, streaming multi-label learning |

----

## Theoritical Studies

| Year       | Venue      | Title  | Remark
| ------------- |:-------------:| --------------:|------------:|
|2019           | ICML  | [Sparse Extreme Multi-label Learning with Oracle Property](http://proceedings.mlr.press/v97/liu19d/liu19d.pdf) | [Code](https://github.com/smloracle/smloracle), by Weiwei Liu |
|2019 | NeurIPS | [Multilabel reductions: what is my loss optimising?](http://papers.nips.cc/paper/9245-multilabel-reductions-what-is-my-loss-optimising.pdf) | [bibtex](https://scholar.googleusercontent.com/scholar.bib?q=info:mz-yIu3FslYJ:scholar.google.com/&output=citation&scisdr=CgUK3ErIELLQ673T7yk:AAGBfm0AAAAAX2LW9ylvLsuvrfJKgZu4PETv4cbbc6GX&scisig=AAGBfm0AAAAAX2LW94UGV94llU318HCTU_i63fA5l1Yw&scisf=4&ct=citation&cd=-1&hl=en), by Google | 

----

## Text Classification

| Year       | Venue     | Title  | Remark
| ------------- |:-------------:| --------------:|------------:|
|2022 | TKDE | [BGNN-XML: Bilateral Graph Neural Networks for Extreme Multi-label Text Classification](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9839555) | |
|2021 | ICML | [SiameseXML: Siamese Networks meet Extreme Classifiers with 100M Labels](http://proceedings.mlr.press/v139/dahiya21a.html) | |
|2020 | KDD | [Correlation Networks for Extreme Multi-label Text Classification](https://dl.acm.org/doi/pdf/10.1145/3394486.3403151?casa_token=Cl5CVnXgiLYAAAAA:e28yssI46chtVhKl7zUWPW0l6q9SPQ7SXRpr5qujghBNsA2iE_s3D2U-q7DNIjSvDAqRNY6wL_OtOw) | [code](https://github.com/XunGuangxu/CorNet) |
|2020 | arXiv | [GNN-XML: Graph Neural Networks for Extreme Multi-label Text Classification](https://www.researchgate.net/profile/Shiliang_Sun2/publication/343537175_GNN-XML_Graph_Neural_Networks_for_Extreme_Multi-label_Text_Classification/links/5f2f9f16299bf13404b13cde/GNN-XML-Graph-Neural-Networks-for-Extreme-Multi-label-Text-Classification.pdf) | |
|2020 | ICML | [Pretrained Generalized Autoregressive Model with Adaptive Probabilistic Label Clusters for Extreme Multi-label Text Classification](https://proceedings.icml.cc/static/paper_files/icml/2020/807-Paper.pdf) | [code](https://github.com/huiyegit/APLC_XLNet) |
|2019 | ACL |[Large-Scale Multi-Label Text Classification on EU Legislation](http://pages.cs.aueb.gr/~rulller/docs/lmtc_eu_Legislation_acl_2019.pdf) | Eur-Lex 4.3K, [bibtex](https://dblp.uni-trier.de/rec/bibtex/conf/acl/ChalkidisFMA19) |
|2019 | arXiv  | [X-BERT: eXtreme Multi-label Text Classification with BERT](https://arxiv.org/pdf/1905.02331.pdf) | [code](https://github.com/OctoberChang/X-BERT) by [Yiming Yang](https://scholar.google.com/citations?hl=en&user=MlZq4XwAAAAJ&view_op=list_works&sortby=pubdate), Inderjit Dhillon |
|2019 | NeurIPS | [AttentionXML: Extreme Multi-Label Text Classification with Multi-Label Attention Based Recurrent Neural Networks](https://arxiv.org/pdf/1811.01727.pdf) |
|2018 | EMNLP | [Few-Shot and Zero-Shot Multi-Label Learning for Structured Label Spaces](https://www.aclweb.org/anthology/D18-1352.pdf) | few-shot, zero-shot, evaluation metric |
|2018 | NeurIPS | A no-regret generalization of hierarchical softmax to extreme multi-label classification | [code](https://github.com/mwydmuch/extremeText), [PLT code](https://github.com/mwydmuch/napkinXC) |
|2017 | SIGIR | [Deep Learning for Extreme Multi-label Text Classification](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf) | by Yiming Yang at CMU, [bibtex](https://dblp.uni-trier.de/rec/bibtex0/conf/sigir/LiuCWY17)|

----

## Others

### Label Correlation
| Year       | Venue     | Title  | Remark
| ------------- |:-------------:| --------------:|------------:|
|2019 | ICML | [DL2: Training and Querying Neural Networks with Logic](http://proceedings.mlr.press/v97/fischer19a/fischer19a.pdf) | |
|2015 | KDD | [Discovering and Exploiting Deterministic Label Relationships in Multi-Label Learning](https://www.researchgate.net/publication/299970511_Discovering_and_Exploiting_Deterministic_Label_Relationships_in_Multi-Label_Learning) | |
|2010 | KDD | [Multi-Label Learning by Exploiting Label Dependency](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.180.4327&rep=rep1&type=pdf) | |

### Long-tailed Continual Learning
| Year       | Venue     | Title  | Remark
| ------------- |:-------------:| --------------:|------------:|
| 2020 | ECCV | [Imbalanced Continual Learning with Partitioning Reservoir Sampling](https://arxiv.org/pdf/2009.03632.pdf) | |

### Train/Test Split
| Year       | Venue     | Title  | Remark
| ------------- |:-------------:| --------------:|------------:|
|2021 | Arxiv | [Stratified Sampling for Extreme Multi-Label Data](https://arxiv.org/pdf/2103.03494.pdf) | | 


### XML Seminar

| Year       | Venue     | Title  | Remark
| ------------- |:-------------:| --------------:|------------:|
|2019 | Dagstuhl Seminar 18291  | [Extreme Classification](http://manikvarma.org/pubs/bengio19.pdf) | |


### Survey References:

1. https://arxiv.org/pdf/1901.00248.pdf
2. http://www.iith.ac.in/~saketha/research/AkshatMTP2018.pdf
3. http://manikvarma.org/pubs/bengio19.pdf
4. [The Emerging Trends of Multi-Label Learning](https://arxiv.org/pdf/2011.11197.pdf)


### XML Datasets [link](https://github.com/DanqingZ/xmc_dataset)

### Extreme Classification Workshops [link](http://manikvarma.org/events/XC20/index.html)
