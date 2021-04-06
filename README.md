This repo pays specially attention to the long-tailed distribution, where labels follow a long-tailed or power-law distribution in the training dataset or/and test dataset. Related papers are sumarized, including its application in computer vision, in particular image classification, and extreme multi-label learning (XML), in particular text categorization.


- Long-tailed Distribution
  * [Long-tailed Distribution in Computer Vision](#Long-tailed-Distribution-in-Computer-Vision)
  * [eXtreme Multi-label Learning](#eXtreme-Multi-label-Learning)
    + [Binary Relevance](#Binary-Relevance)
    + [Tree-based Methods](#Tree-based-Methods)
    + [Embedding-based Methods](#Embedding-based-Methods)
    + [Speed-up and Compression](#Speed-up-and-Compression)
    + [Noval XML Setups](#Noval-XML-Settings)
    + [Theoritical Studies](#Theoritical-Studies)
    + [Text Classification](#Text-Classification)
    + [Others](#Others)
  


<!-- toc -->


# Long-tailed Distribution in Computer Vision

| Year       | Venue     | Title  | Remark
| ------------- |:-------------:| --------------:|------------:|
|2021 | CVPR | [CReST: A Class-Rebalancing Self-Training Framework for Imbalanced Semi-Supervised Learning](https://arxiv.org/pdf/2102.09559.pdf) | by Google |
|2021 | ICLR | [LONG-TAILED RECOGNITION BY ROUTING DIVERSE DISTRIBUTION-AWARE EXPERTS](https://arxiv.org/pdf/2010.01809.pdf) | by Zi-Wei Liu |
|2021 | ICLR | [Long-Tail Learning via Logit Adjustment](https://arxiv.org/pdf/2007.07314.pdf) | by Google |
|2020 | CVPR | [Equalization Loss for Long-Tailed Object Recognition](https://arxiv.org/pdf/2003.05176.pdf) | |
|2020 | ICLR | [Decoupling representation and classifier for long-tailed recognition](https://openreview.net/pdf?id=r1gRTCVFvB) | [Code](https://github.com/facebookresearch/classifier-balancing) |
|2020 | CVPR | [Bbn: Bilateral-branch network with cumulative learning for long-tailed visual recognition](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_BBN_Bilateral-Branch_Network_With_Cumulative_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2020_paper.pdf) | [Code](https://github.com/Megvii-Nanjing/BBN) |
|2021           | Arxiv | [Learning From Multiple Experts: Self-paced Knowledge Distillation for Long-tailed Classification](https://arxiv.org/pdf/2001.01536.pdf) | |
|2019 | NIPS | [Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss](https://arxiv.org/pdf/1906.07413.pdf) | [Code](https://github.com/kaidic/LDAM-DRW) |
|2019           | CVPR  | [Large-Scale Long-Tailed Recognition in an Open World](https://arxiv.org/pdf/1904.05160.pdf) | [Code](https://github.com/smloracle/smloracle), [bibtex](https://dblp.uni-trier.de/rec/bibtex/conf/cvpr/0002MZWGY19), by CUHK |
|2018 | - | [iNatrualist. The inaturalist 2018 competition dataset](https://github.com/visipedia/inat_comp/tree/master/2018) | long-tailed dataset |
|2017 | Arxiv | [The Devil is in the Tails: Fine-grained Classification in the Wild](https://arxiv.org/pdf/1709.01450.pdf) | |
|2017 | NIPS | Learning to model the tail | |

----



# eXtreme Multi-label Learning


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
