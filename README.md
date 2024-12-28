# Label-Efficient Federated Learning

This is the repository for **Towards Label-Efficient Federated Learning: Taxonomy, Review, and Emerging Trends**, a curated list of recent advancements in federated learning (FL) methods for label-efficient learning.

We will update the list of papers regularly to keep it up to date. :grin:

------

# Menu

- [Inaccurate Supervision in FL](#inaccurate-supervision-in-fl)
  - [Federated Noisy Learning](#federated-noisy-learning)
    - [Sample-Level](#sample-level)
    - [Client-Level](#client-level)
- [Incomplete Supervision in FL](#incomplete-supervision-in-fl)
  - [Federated Active Learning](#federated-active-learning)
    - [Pool-Based](#pool-based)
    - [Stream-Based](#stream-based)
  - [Federated Semi-Supervised Learning](#federated-semi-supervised-learning)
    - [Label at All Clients](#label-at-all-clients)
    - [Label at Partial Clients](#label-at-partial-clients)
    - [Label at Server](#label-at-server)
    - [Unlabeled at Server](#unlabeled-at-server)
- [No Supervision in FL](#no-supervision-in-fl)
  - [Federated Unsupervised Learning](#federated-unsupervised-learning)
    - [Clustering-Based](#clustering-based)
    - [Generative Methods](#generative-methods)
  - [Federated Self-Supervised Learning](#federated-self-supervised-learning)
    - [Contrastive Learning](#contrastive-learning)

------

# Inaccurate Supervision in FL

## Federated Noisy Learning

### Sample-Level

- [ IEEE Intelligent Systems-2022 ] Robust Federated Learning With Noisy Labels [[paper](https://ieeexplore.ieee.org/abstract/document/9713942)]

- [ CIKM-2022 ] Towards Federated Learning against Noisy Labels via Local Self-Regularization [[paper](https://dl.acm.org/doi/pdf/10.1145/3511808.3557475)] [[code](https://github.com/Sprinter1999/FedLSR)]

- [ ECAI-2023 ] FedCoop: Cooperative Federated Learning for Noisy Labels [[paper](https://ebooks.iospress.nl/doi/10.3233/FAIA230529)]

- [ DASFAA-2023 ] A Static Bi-dimensional Sample Selection for Federated Learning with Label Noise [[paper](https://link.springer.com/chapter/10.1007/978-3-031-30637-2_49)]

- [ AAAI-2024 ] FedFixer: Mitigating Heterogeneous Label Noise in Federated Learning [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29179)]

- [ ICASSP-2024 ] Federated Learning with Instance-Dependent Noisy Label [[paper](https://ieeexplore.ieee.org/abstract/document/10447823)] [[code](https://github.com/TriStonesWang/FedBeat)]

- [ AAAI-2024 ] FedDiv: Collaborative Noise Filtering for Federated Learning with Noisy Labels [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28095)] [[code](https://github.com/lijichang/FLNL-FedDiv)]

- [ AAAI-2024 ] Federated Label-Noise Learning with Local Diversity Product Regularization [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29659)]

### Client-Level

- [ IEEE TVT-2022 ] Client Selection for Federated Learning With Label Noise [[paper](https://ieeexplore.ieee.org/abstract/document/9632344)]

- [ CVPR-2022 ] Robust Federated Learning With Noisy and Heterogeneous Clients [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Fang_Robust_Federated_Learning_With_Noisy_and_Heterogeneous_Clients_CVPR_2022_paper.pdf)] [[code](https://github.com/xiye7lai/RHFL)]

- [ CVPR-2022 ] FedCorr: Multi-Stage Federated Learning for Label Noise Correction [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_FedCorr_Multi-Stage_Federated_Learning_for_Label_Noise_Correction_CVPR_2022_paper.pdf)] [[code](https://github.com/Xu-Jingyi/FedCorr)]

- [ CIKM-2022 ] FedRN: Exploiting k-Reliable Neighbors Towards Robust Federated Learning  [[paper](https://dl.acm.org/doi/abs/10.1145/3511808.3557322)] [[code](https://github.com/ElvinKim/FedRN)]

- [ IJCAI-2023 ] FedNoRo: Towards Noise-Robust Federated Learning by Addressing Class Imbalance and Label Noise Heterogeneity [[paper](https://www.ijcai.org/proceedings/2023/0492.pdf)] [[code](https://github.com/wnn2000/FedNoRo)]

- [ AAAI-2024 ] Federated Learning with Extremely Noisy Clients via Negative Distillation [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29329)] [[code](https://github.com/linChen99/FedNed)]

- [ CIKM-2024 ] Tackling Noisy Clients in Federated Learning with End-to-End Label Correction [[paper](https://dl.acm.org/doi/pdf/10.1145/3627673.3679550)] [[code](https://github.com/Sprinter1999/FedELC)]

- [ AAAI-2024 ] FedA3I: Annotation Quality-Aware Aggregation for Federated Medical Image Segmentation against Heterogeneous Annotation Noise [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29525)] [[code](https://github.com/wnn2000/FedAAAI)]

- [ TMC-2024 ] Overcoming Noisy Labels and Non-IID Data in Edge Federated Learning [[paper](https://ieeexplore.ieee.org/document/10526454)]

------

# Incomplete Supervision in FL

## Federated Active Learning

### Pool-Based

- [ IEEE Access-2020 ] Active Learning Based Federated Learning for Waste and Natural Disaster Image Classification [[paper](https://ieeexplore.ieee.org/abstract/document/9261337)]

- [ IEEE Access-2024 ] Federated Active Learning (F-AL): An Efficient Annotation Strategy for Federated Learning [[paper](https://ieeexplore.ieee.org/abstract/document/10471526)]

- [ ACM MM-2023 ] AffectFAL: Federated Active Affective Computing with Non-IID Data [[paper](https://dl.acm.org/doi/pdf/10.1145/3581783.3612442)] [[code](https://github.com/AffectFAL/AffectFAL)]

- [ CVPR-2023 ] Re-Thinking Federated Active Learning Based on Inter-Class Diversity [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Kim_Re-Thinking_Federated_Active_Learning_Based_on_Inter-Class_Diversity_CVPR_2023_paper.pdf)] [[code](https://github.com/raymin0223/LoGo)]

- [ ICCV-2023 ] Knowledge-Aware Federated Active Learning with Non-IID Data [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_Knowledge-Aware_Federated_Active_Learning_with_Non-IID_Data_ICCV_2023_paper.pdf)] [[code](https://github.com/ycao5602/KAFAL)]

- [ CVPR-2024 ] Think Twice Before Selection: Federated Evidential Active Learning for Medical Image Analysis with Domain Shifts [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Think_Twice_Before_Selection_Federated_Evidential_Active_Learning_for_Medical_CVPR_2024_paper.pdf)] [[code](https://github.com/JiayiChen815/FEAL)]

### Stream-Based

- [ arXiv-2020 ] Combining Federated and Active Learning for Communication-Efficient Distributed Failure Prediction in Aeronautics [[paper](https://arxiv.org/pdf/2001.07504)]

- [ Personal and Ubiquitous Computing-2022 ] Semi-Supervised and personalized federated activity recognition based on active learning and label propagation [[paper](https://link.springer.com/content/pdf/10.1007/s00779-022-01688-8.pdf)]

- [ TMC-2024 ] AffectFAL: Federated Active Affective Computing with Non-IID Data [[paper](https://ieeexplore.ieee.org/abstract/document/10266762)]

- [ AAAI-2025 ] Learn How to Query from Unlabeled Data Streams in Federated Learning [[paper](https://arxiv.org/pdf/2412.08138)] [[code](https://github.com/hiyuchang/leadq)]

## Federated Semi-Supervised Learning

### Label at All Clients

- [ ICLR-2021 ] Federated Semi-Supervised Learning with Inter-Client Consistency & Disjoint Learning [[paper](https://openreview.net/pdf?id=ce6CFXBh30h)] [[code](https://github.com/wyjeong/FedMatch)]

- [ arXiv-2021 ] SemiFed: Semi-Supervised Federated Learning with Consistency and Pseudo-Labeling [[paper](https://arxiv.org/pdf/2108.09412)]

- [ IEEE BigData-2021 ] FedTriNet: A Pseudo Labeling Method with Three Players for Federated Semi-supervised Learning [[paper](https://ieeexplore.ieee.org/abstract/document/9671374)] [[code](https://github.com/Michelingweo/FedTriNet)]

- [ TNNLS-2024 ] Balanced Federated Semisupervised Learning With Fairness-Aware Pseudo-Labeling [[paper](https://ieeexplore.ieee.org/abstract/document/10008103)]

- [ CIKM-2023 ] Non-IID always Bad? Semi-Supervised Heterogeneous Federated Learning with Local Knowledge Enhancement [[paper](https://dl.acm.org/doi/pdf/10.1145/3583780.3614991)] [[code](https://github.com/zcfinal/FedLoKe)]

- [ ICCV-2023 ] Local or Global: Selective Knowledge Assimilation for Federated Learning with Limited Labels [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Cho_Local_or_Global_Selective_Knowledge_Assimilation_for_Federated_Learning_with_ICCV_2023_paper.pdf)]

- [ ICDE-2024 ] Clients Help Clients: Alternating Collaboration for Semi-Supervised Federated Learning [[paper](https://ieeexplore.ieee.org/abstract/document/10598007)]

- [ AAAI-2024 ] Combating Data Imbalances in Federated Semi-Supervised Learning with Dual Regulators [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28974)] [[code](https://github.com/White1973/FedDure)]

- [ IJCAI-2024 ] Estimating before Debiasing: A Bayesian Approach to Detaching Prior Bias in Federated Semi-Supervised Learning [[paper](https://www.ijcai.org/proceedings/2024/0290.pdf)] [[code](https://github.com/GuogangZhu/FedDB)]

### Label at Partial Clients

- [ Medical Image Analysis-2021 ] Federated semi-Supervised learning for COVID region segmentation in chest CT using multi-national data from China, Italy, Japan [[paper](https://www.sciencedirect.com/science/article/pii/S1361841521000384)]

- [ MICCAI-2021 ] Federated Semi-Supervised Medical Image Classification via Inter-client Relation Matching [[paper](https://link.springer.com/chapter/10.1007/978-3-030-87199-4_31)] [[code](https://github.com/liuquande/FedIRM)]

- [ CVPR-2022 ] RSCFed: Random Sampling Consensus Federated Semi-Supervised Learning [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liang_RSCFed_Random_Sampling_Consensus_Federated_Semi-Supervised_Learning_CVPR_2022_paper.pdf)] [[code](https://github.com/xmed-lab/RSCFed)]

- [ CVPR-2023 ] Class Balanced Adaptive Pseudo Labeling for Federated Semi-Supervised Learning [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Class_Balanced_Adaptive_Pseudo_Labeling_for_Federated_Semi-Supervised_Learning_CVPR_2023_paper.pdf)] [[code](https://github.com/minglllli/CBAFed)]

- [ AAAI-2024 ] FedCD: Federated Semi-Supervised Learning with Class Awareness Balance via Dual Teachers [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28175)] [[code](https://github.com/YunzZ-Liu/FedCD)]

- [ ICLR-2024 ] Robust Training of Federated Models with Extremely Label Deficiency [[paper](https://openreview.net/pdf?id=qxLVaYbsSI)] [[code](https://github.com/tmlr-group/Twin-sight)]

- [ TMC-2024 ] Dual Class-Aware Contrastive Federated Semi-Supervised Learning [[paper](https://ieeexplore.ieee.org/abstract/document/10705896)]

### Label at Server

- [ ICLR-2021 ] Federated Semi-Supervised Learning with Inter-Client Consistency & Disjoint Learning [[paper](https://openreview.net/pdf?id=ce6CFXBh30h)] [[code](https://github.com/wyjeong/FedMatch)]

- [ arXiv-2021 ] FedCon: A Contrastive Framework for Federated Semi-Supervised Learning [[paper](https://arxiv.org/pdf/2109.04533)] [[code](https://github.com/zewei-long/fedcon-pytorch)]

- [ IEEE BigData-2021 ] Improving Semi-Supervised Federated Learning by Reducing the Gradient Diversity of Models [[paper](https://ieeexplore.ieee.org/abstract/document/9671693)]

- [ TITS-2022 ] Semi-Supervised Federated Learning for Travel Mode Identification From GPS Trajectories [[paper](https://ieeexplore.ieee.org/abstract/document/9514368)]

- [ NeurIPS-2022 ] SemiFL: Semi-Supervised Federated Learning for Unlabeled Clients with Alternate Training [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/71c3451f6cd6a4f82bb822db25cea4fd-Paper-Conference.pdf)] [[code](https://github.com/diaoenmao/SemiFL-Semi-Supervised-Federated-Learning-for-Unlabeled-Clients-with-Alternate-Training)]

- [ SDM-2023 ] Knowledge-Enhanced Semi-Supervised Federated Learning for Aggregating Heterogeneous Lightweight Clients in IoT [[paper](https://epubs.siam.org/doi/epdf/10.1137/1.9781611977653.ch56)] [[code](https://github.com/JackqqWang/pfedknow/tree/master)]

- [ arXiv-2024 ] FedAnchor: Enhancing Federated Semi-Supervised Learning with Label Contrastive Loss for Unlabeled Clients [[paper](https://arxiv.org/pdf/2402.10191)]

- [ NeurIPS-2024 ] (FL)$^2$: Overcoming Few Labels in Federated Semi-Supervised Learning [[paper](https://openreview.net/pdf?id=lflwtGE6Vf)] [[code](https://github.com/seungjoo-ai/FLFL-NeurIPS24)]

### Unlabeled at Server

- [ ICLR-2017 ] Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data [[paper](https://openreview.net/pdf?id=HkwoSDPgg)] [[code](https://github.com/tensorflow/privacy/tree/master/research/pate_2017)]

- [ ICLR-2018 ] Scalable Private Learning with PATE [[paper](https://openreview.net/pdf?id=rkZB1XbRZ)] [[code](https://github.com/tensorflow/privacy/tree/master/research/pate_2018)]

- [ ICDE-2022 ] Enhancing Federated Learning with In-Cloud Unlabeled Data [[paper](https://ieeexplore.ieee.org/abstract/document/9835163)]

- [ TMC-2024 ] Enhancing Federated Learning With Server-Side Unlabeled Data by Adaptive Client and Data Selection [[paper](https://ieeexplore.ieee.org/abstract/document/10094013)]

------

# No Supervision in FL

## Federated Unsupervised Learning

### Clustering-Based

- [ ICML-2022 ] Orchestra: Unsupervised Federated Learning via Globally Consistent Clustering [[paper](https://proceedings.mlr.press/v162/lubana22a/lubana22a.pdf)] [[code](https://github.com/akhilmathurs/orchestra)]

- [ arXiv-2024 ] Fair Federated Data Clustering through Personalization: Bridging the Gap between Diverse Data Distributions [[paper](https://arxiv.org/pdf/2407.04302)] [[code](https://github.com/P-FClus/p-FClus)]

- [ ACM MM-2024 ] Heterogeneity-Aware Federated Deep Multi-View Clustering towards Diverse Feature Representations [[paper](https://dl.acm.org/doi/pdf/10.1145/3664647.3681302)] [[code](https://github.com/xiaorui-jiang/HFMVC)]

### Generative Methods

- [ arXiv-2023 ] UFed-GAN: A Secure Federated Learning Framework with Constrained Computation and Unlabeled Data [[paper](https://arxiv.org/pdf/2308.05870)]

- [ arXiv-2023 ] PFL-GAN: When Client Heterogeneity Meets Generative Models in Personalized Federated Learning [[paper](https://arxiv.org/pdf/2308.12454)]

## Federated Self-Supervised Learning

### Contrastive Learning

- [ FITEE-2023 ] Federated Unsupervised Representation Learning [[paper](https://www.fitee.zjujournals.com/en/article/doi/10.1631/FITEE.2200268/)]

- [ ICCV-2021 ] Collaborative Unsupervised Visual Representation Learning from Decentralized Data [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhuang_Collaborative_Unsupervised_Visual_Representation_Learning_From_Decentralized_Data_ICCV_2021_paper.pdf)] 

- [ arXiv-2021 ] SSFL: Tackling Label Deficiency in Federated Learning via Personalized Self-Supervision [[paper](https://arxiv.org/pdf/2110.02470)] [[code](https://openreview.net/attachment?id=y1faDxZ_-0a&name=supplementary_material)]

- [ ICLR-2022 ] Divergence-aware Federated Self-Supervised Learning [[paper](https://openreview.net/pdf?id=oVE1z8NlNe)]

- [ ECCV-2022 ] FedX: Unsupervised Federated Learning with Cross Knowledge Distillation [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Rehman_L-DAWA_Layer-wise_Divergence_Aware_Weight_Aggregation_in_Federated_Self-Supervised_Visual_ICCV_2023_paper.pdf)] [[code](https://github.com/Sungwon-Han/FEDX)]

- [ ICCV-2023 ] L-DAWA: Layer-wise Divergence Aware Weight Aggregation in Federated Self-Supervised Visual Representation Learning [[paper](https://ieeexplore.ieee.org/abstract/document/10377103)] [[code](https://github.com/yasar-rehman/L-DAWA)]

- [ ICCV-2023 ] ProtoFL: Unsupervised Federated Learning via Prototypical Distillation [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Kim_ProtoFL_Unsupervised_Federated_Learning_via_Prototypical_Distillation_ICCV_2023_paper.pdf)]

- [ CVPR-2024 ] Rethinking the Representation in Federated Unsupervised Learning with Non-IID Data [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Liao_Rethinking_the_Representation_in_Federated_Unsupervised_Learning_with_Non-IID_Data_CVPR_2024_paper.pdf)] [[code](https://github.com/XeniaLLL/FedU2)]

- [ IEEE TCCN-2024 ] Unsupervised Federated Optimization at the Edge: D2D-Enabled Learning Without Labels [[paper](https://ieeexplore.ieee.org/abstract/document/10507170)]
