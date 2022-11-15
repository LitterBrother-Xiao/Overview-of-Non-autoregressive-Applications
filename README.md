# Overview-of-Non-autoregressive-Applications
This repo presents an overview of Non-autoregressive (NAR) models, including links to related papers and corresponding codes.

NAR models aim to speed up decoding and reduce the inference latency, then realize better industry application. However, this improvement of speed comes at the expense of the decline of quality. Many methods and tricks are proposed to reduce this gap.

NAR models are first proposed for neural machine translation, and then are applied for various tasks, such as speech to text, speech gneration, speech translation, text summarization; dialogue and intent detection; grammatical error correction; text style transfer; semantic parsing and etc.

A survey on non-autoregressive neural machine translation including a brief review of other various tasks can be found on https://arxiv.org/abs/2204.09269.

## Neural machine translation
### Tutorial
ACL 2022 [Non-Autoregressive Sequence Generation](https://github.com/NAR-tutorial/acl2022)
### Papers
### Data manipulation
### &nbsp;&nbsp;&nbsp;&nbsp; Knowledge distillation
- [20ICLR] UNDERSTANDING KNOWLEDGE DISTILLATION IN NON-AUTOREGRESSIVE MACHINE TRANSLATION. [Paper](https://arxiv.org/pdf/1911.02727.pdf.) &
[Code](https://github.com/pytorch/fairseq/tree/master/examples/nonautoregressive_translation.)  
- [20ACL] A Study of Non-autoregressive Model for Sequence Generation. [Paper](https://aclanthology.org/2020.acl-main.15.pdf)  
- [20ACL] Improving Non-autoregressive Neural Machine Translation with Monolingual Data. [paper](https://aclanthology.org/2020.acl-main.171.pdf) 
- [21ICLR] UNDERSTANDING AND IMPROVING LEXICAL CHOICE IN NON-AUTOREGRESSIVE TRANSLATION. [Paper](https://arxiv.org/pdf/2012.14583v2.pdf)  
- [21ACL-IJCNLP] Rejuvenating Low-Frequency Words: Making the Most of Parallel Data in Non-Autoregressive Translation. [Paper](https://arxiv.org/pdf/2106.00903.pdf) & [Code](https://github.com/longyuewangdcu/RLFW-NAT)  
- [22Findings of ACL-IJCNLP] How Does Distilled Data Complexity Impact the Quality and Confidence of Non-Autoregressive Machine Translation? [Paper](https://aclanthology.org/2021.findings-acl.385.pdf)
- [UnderReview] Self-Distillation Mixup Training for Non-autoregressive Neural Machine Translation. [Paper](https://arxiv.org/pdf/2112.11640v1.pdf)  
- [22NAACL] Neighbors Are Not Strangers: Improving Non-Autoregressive Translation under Low-Frequency Lexical Constraints. [Paper](https://openreview.net/pdf?id=T-Wh9Ds-qk)
- [22ArXiv] Speed Up Iterative Non-Autoregressive Transformers by Distilling Multiple Steps. [Paper](https://arxiv.org/pdf/2206.02999.pdf) & [Code](https://github.com/layer6ai-labs/DiMS)

### &nbsp;&nbsp;&nbsp;&nbsp; Data learning strategy 
- [21ACL-IJCNLP] Glancing Transformer for Non-Autoregressive Neural Machine Translation. [Paper](https://aclanthology.org/2021.acl-long.155.pdf) & [Code](https://github.com/FLC777/GLAT)  
- [21ACL-IJCNLP Findinds] Progressive Multi-Granularity Training for Non-Autoregressive Translation. [Paper](https://aclanthology.org/2021.findings-acl.247.pdf) 
- [21ArXiv] MvSR-NAT: Multi-view Subset Regularization for Non-Autoregressive Machine Translation. [Paper](https://arxiv.org/pdf/2108.08447.pdf)  
- [22ACL] latent-GLAT: Glancing at Latent Variables for Parallel Text Generation. [Paper](https://arxiv.org/pdf/2204.02030.pdf) & [Code](https://github.com/baoy-nlp/Latent-GLAT)
- [22NAACL] Non-Autoregressive Neural Machine Translation with Consistency Regularization Optimized Variational Framework. [Paper](https://openreview.net/pdf?id=cLe29FcNAKb)
- [22ArXiv] Contrastive Conditional Masked Language Model for Non-autoregressive Neural Machine Translation. [Paper](https://openreview.net/pdf?id=9_j8yJ6ISSr)

### Modeling 
### &nbsp;&nbsp;&nbsp;&nbsp;Iteration-based methods
- [18EMNLP] Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement. [Paper](https://aclanthology.org/D18-1149.pdf) & [Code](https://github.com/nyu-dl/dl4mt-nonauto)   
- [19ICML] Insertion Transformer: Flexible Sequence Generation via Insertion Operations. [Paper](https://arxiv.org/pdf/1902.03249.pdf) & [Code](https://github.com/pytorch/fairseq/tree/master/examples/nonautoregressive_translation)   
- [19NeurIPS] Levenshtein Transformer. [Paper](https://arxiv.org/pdf/1905.11006v1.pdf) & [Code](https://github.com/pytorch/fairseq/tree/master/examples/nonautoregressive_translation)  
- [19EMNLP-IJCNLP] Mask-Predict: Parallel Decoding of Conditional Masked Language Models. [Paper](https://aclanthology.org/D19-1633.pdf) & [Code](https://github.com/facebookresearch/Mask-Predict)  
- [20ArXiv] Semi-autoregressive training improves mask-predict decoding. [Paper](https://arxiv.org/pdf/2001.08785.pdf)  
- [20ICML] Non-autoregressive Machine Translation with Disentangled Context Transformer. [Paper](https://arxiv.org/pdf/2001.05136.pdf) & [Code](https://github.com/facebookresearch/DisCo)  
- [20EMNLP] Semi-Autoregressive Neural Machine Translation. [Paper](https://aclanthology.org/D18-1044.pdf)   
- [20ACL] ENGINE: Energy-Based Inference Networks for Non-Autoregressive Machine Translation. [Paper](https://aclanthology.org/2020.acl-main.251.pdf) & [Code](https://github.com/lifu-tu/ENGINE)    
- [20ACL] Jointly Masked Sequence-to-Sequence Model for Non-Autoregressive Neural Machine Translation. [Paper](https://aclanthology.org/2020.acl-main.36.pdf) & [Code](https://github.com/lemmonation/jm-nat)  
- [20ICLR] DEEP ENCODER, SHALLOW DECODER:REEVALUATING NON-AUTOREGRESSIVE MACHINE TRANSLATION. [Ppaer](https://arxiv.org/pdf/2006.10369.pdf) & [Code](https://github.com/jungokasai/deep-shallow)  
- [21EMNLP] Learning to Rewrite for Non-Autoregressive Neural Machine Translation. [Paper](https://aclanthology.org/2021.emnlp-main.265.pdf) & [Code](https://github.com/xwgeng/RewriteNAT)  
- [22ICLR] STEP-UNROLLED DENOISING AUTOENCODERS FOR TEXT GENERATION. [Paper](https://arxiv.org/pdf/2112.06749.pdf) & [Code](https://github.com/vvvm23/sundae)
- [22ICLR] IMPROVING NON-AUTOREGRESSIVE TRANSLATION MODELS WITHOUT DISTILLATION. [Paper](https://openreview.net/pdf?id=I2Hw58KHp8O) & [Code](https://github.com/layer6ai-labs/CMLMC) 
- [UnderReview] DEEP EQUILIBRIUM NON-AUTOREGRESSIVE SEQUENCE LEARNING. [Paper](https://openreview.net/pdf?id=bnkvnbGEXnc) 
- [22ArXiv] Non-Autoregressive Machine Translation with Translation Memories. [Paper](https://arxiv.org/pdf/2210.06020.pdf)
- [22ArXiv] Nearest Neighbor Non-autoregressive Text Generation. [Paper](https://arxiv.org/pdf/2208.12496.pdf)
- [22NeurIPS] INSNET: An Efficient, Flexible, and Performant Insertion-based Text Generation Model. [Paper](file:///G:/2022-1-24/%E6%95%B0%E6%8D%AE/nmt/2022/neuips_iclr/INSNETAn%20Efficient,%20Flexible,%20and%20Performant.pdf)

### &nbsp;&nbsp;&nbsp;&nbsp;Latent variable-based methods
- [18ICLR] NON-AUTOREGRESSIVE NEURAL MACHINE TRANSLATION. [Paper](https://arxiv.org/pdf/1711.02281.pdf) & [Code](https://github.com/salesforce/nonauto-nmt)
- [19EMNLP-IJCNLP] FlowSeq: Non-Autoregressive Conditional Sequence Generation with Generative Flow. [Paper](https://arxiv.org/pdf/1909.02480v1.pdf) & [Code](https://github.com/XuezheMax/flowseq)
- [19NeurIPS] Fast Structured Decoding for Sequence Models. [Paper](https://arxiv.org/pdf/1910.11555.pdf) 
- [19ArXiv] Non-autoregressive Transformer by Position Learning. [Paper](https://arxiv.org/pdf/1911.10677.pdf)  
- [19ACL] Syntactically Supervised Transformers for Faster Neural Machine Translation. [Paper](https://aclanthology.org/P19-1122.pdf) & [Code](https://github.com/dojoteef/synst)
- [20AAAI] Latent-Variable Non-Autoregressive Neural Machine Translation with Deterministic Inference using a Delta Posterior. [Paper](https://arxiv.org/pdf/1908.07181v1.pdf) & [Code](https://github.com/zomux/lanmt)   
- [20EMNLP] Non-Autoregressive Machine Translation with Latent Alignments. [Paper](https://aclanthology.org/2020.emnlp-main.83.pdf)  
- [20ArXiv] Incorporating a Local Translation Mechanism into Non-autoregressive Translation. [Paper](https://arxiv.org/pdf/2011.06132.pdf) & [Code](https://github.com/shawnkx/NAT-with-Local-AT)
- [21EMNLP] AligNART: Non-autoregressive Neural Machine Translation by Jointly Learning to Estimate Alignment and Translate. [Paper](https://aclanthology.org/2021.emnlp-main.1.pdf)  
- [21EACL] Enriching Non-Autoregressive Transformer with Syntactic and Semantic Structures for Neural Machine Translation. [Paper](https://aclanthology.org/2021.eacl-main.105.pdf)   
- [21AAAI] Guiding Non-Autoregressive Neural Machine Translation Decoding with Reordering Information. [Paper](https://arxiv.org/pdf/1911.02215.pdf) & [Code](https://github.com/ranqiu92/ReorderNAT)  
- [21NAACL-HLT] Non-Autoregressive Translation by Learning Target Categorical Codes. [Paper](https://aclanthology.org/2021.naacl-main.458.pdf) & [Code](https://github.com/baoy-nlp/CNAT)  
- [21ACL-IJCNLP Findinds] Fully Non-autoregressive Neural Machine Translation:Tricks of the Trade. [Paper](https://aclanthology.org/2021.findings-acl.11.pdf) & [Code](https://github.com/pytorch/fairseq/tree/main/examples/nonautoregressive_translation)  

### &nbsp;&nbsp;&nbsp;&nbsp;Other enhancements-based mothods
- [19AAAI] Non-Autoregressive Neural Machine Translation with Enhanced Decoder Input. [Paper](https://arxiv.org/pdf/1812.09664.pdf)  
- [19AAAI] Non-Autoregressive Machine Translation with Auxiliary Regularization. [Paper](https://arxiv.org/pdf/1902.10245.pdf)   
- [20COLING] Context-Aware Cross-Attention for Non-Autoregressive Translation. [Paper](https://aclanthology.org/2020.coling-main.389.pdf)  
- [21ArXiv] LAVA NAT: A Non-Autoregressive Translation Model with Look-Around Decoding and Vocabulary Attention. [Paper](https://arxiv.org/pdf/2002.03084v1.pdf)
- [21AAAI] Non-Autoregressive Translation with Layer-Wise Prediction and Deep Supervision. [Paper](https://arxiv.org/abs/2110.07515) & [Code](https://github.com/chenyangh/DSLP)
- [22ArXiv] Non-autoregressive Translation with Dependency-Aware Decoder. [Paper](https://arxiv.org/pdf/2203.16266.pdf) & [Code](https://github.com/zja-nlp/NAT_with_DAD)
- [22ICML] Directed Acyclic Transformer for Non-Autoregressive Machine Translation. [Paper](https://arxiv.org/pdf/2205.07459.pdf)
- [UnderReview] FUZZY ALIGNMENTS IN DIRECTED ACYCLIC GRAPH FOR NON-AUTOREGRESSIVE MACHINE TRANSLATION. [Paper](https://openreview.net/pdf?id=LSz-gQyd0zE) 
- [22EMNLP findings] Viterbi Decoding of Directed Acyclic Transformer for Non-Autoregressive Machine Translation. [Paper](https://arxiv.org/pdf/2210.05193.pdf) 

### Criterion
- [06ICML] Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks. [Paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf) & [Code](https://github.com/parlance/ctcdecode)  
- [18EMNLP] End-to-End Non-Autoregressive Neural Machine Translation with Connectionist Temporal Classification. [Paper](https://aclanthology.org/D18-1336.pdf)  
- [19ACL] Retrieving Sequential Information for Non-Autoregressive Neural Machine Translation. [Paper](https://aclanthology.org/P19-1288.pdf) & [Code](https://github.com/ictnlp/RSI-NAT)  
- [20AAAI] Minimizing the Bag-of-Ngrams Difference for Non-Autoregressive Neural Machine Translation. [Paper](https://arxiv.org/pdf/1911.09320.pdf) & [Code](https://github.com/ictnlp/BoN-NAT)  
- [20ICML] Aligned Cross Entropy for Non-Autoregressive Machine Translation. [Paper](https://arxiv.org/pdf/2004.01655.pdf) & [Code](https://github.com/m3yrin/aligned-cross-entropy)  
- [21ICML] Order-Agnostic Cross Entropy for Non-Autoregressive Machine Translation. [Paper](https://arxiv.org/pdf/2106.05093.pdf) & [Code](https://github.com/tencent-ailab/ICML21_OAXE)
- [22ICML] On the Learning of Non-Autoregressive Transformers. [Paper](https://arxiv.org/pdf/2206.05975.pdf)
- [22NAACL] One Reference Is Not Enough: Diverse Distillation with Reference Selection for Non-Autoregressive Translation. [Paper](https://arxiv.org/pdf/2205.14333.pdf) & [Code](https://github.com/ictnlp/DDRS-NAT)
- [22EMNLP] Multi-Granularity Optimization for Non-Autoregressive Translation. [Paper](https://arxiv.org/pdf/2210.11017.pdf)
- [22COLNG] ngram-OAXE: Phrase-Based Order-Agnostic Cross Entropy for Non-Autoregressive Machine Translation. [Paper](https://arxiv.org/pdf/2210.03999.pdf) & [Code]( https://github.com/tencent-ailab/machine-translation/COLING22_ngram-OAXE/)
- [22NeurIPS] Non-Monotonic Latent Alignments for CTC-Based Non-Autoregressive Machine Translation. [Paper](https://arxiv.org/pdf/2210.03953.pdf) & 
[Code](https://github.com/ictnlp/NMLA-NAT.)
- [22NAACL] A Study of Syntactic Multi-Modality in Non-Autoregressive Machine Translation. [Paper](https://arxiv.org/pdf/2207.04206.pdf)

### Decoding
- [18EMNLP] Semi-Autoregressive Neural Machine Translation. [Paper](https://aclanthology.org/D18-1044.pdf)   
- [20ACL] Learning to Recover from Multi-Modality Errors for Non-Autoregressive Neural Machine Translation. [Paper](https://aclanthology.org/2020.acl-main.277.pdf) & [Code](https://github.com/ranqiu92/RecoverSAT) 
- [20COLING] Train Once, and Decode As You Like. [Paper](https://aclanthology.org/2020.coling-main.25.pdf)
- [UnderReview] Diformer: Directional Transformer for Neural Machine Translation. [Paper](https://arxiv.org/pdf/2112.11632v2.pdf)
- [22ArXiv] Lossless Speedup of Autoregressive Translation with Generalized Aggressive Decoding. [Paper](https://arxiv.org/pdf/2203.16487v2.pdf) & [Code](https://github.com/hemingkx/Generalized-Aggressive-Decoding)
- [UnderReview] HYBRID-REGRESSIVE NEURAL MACHINE TRANSLATION. [Paper](https://openreview.net/pdf?id=2NQ8wlmU9a_)

### Benefiting from Pre-trained Modoels
- [19ACL] Imitation Learning for Non-Autoregressive Neural Machine Translation. [Paper](https://aclanthology.org/P19-1125.pdf)  
- [19EMNLP-IJCNLP] Hint-Based Training for Non-Autoregressive Machine Translation. [Paper](https://aclanthology.org/D19-1573.pdf) & [Code](https://github.com/zhuohan123/hint-nart)  
- [20AAAI] Fine-Tuning by Curriculum Learning for Non-Autoregressive Neural Machine Translation. [Paper](http://staff.ustc.edu.cn/~linlixu/papers/aaai20a.pdf) & [Code](https://github.com/lemmonation/fcl-nat)  
- [20ICML] An EM Approach to Non-autoregressive Conditional Sequence Generation. [Paper](https://arxiv.org/pdf/2006.16378.pdf) & [Code](https://github.com/Edward-Sun/NAT-EM)  
- [20ACL] ENGINE: Energy-Based Inference Networks for Non-Autoregressive Machine Translation. [Paper](https://aclanthology.org/2020.acl-main.251.pdf) & [Code]( https://github.com/lifu-tu/ENGINE)
- [20AutoSimtrans] Improving Autoregressive NMT with Non-Autoregressive Model. [Paper](https://aclanthology.org/2020.autosimtrans-1.4.pdf)  
- [21IJCAI] Task-Level Curriculum Learning for Non-Autoregressive Neural Machine Translation. [Paper](https://www.ijcai.org/Proceedings/2020/0534.pdf)  
- [21NAACL-HLT] Multi-Task Learning with Shared Encoder for Non-Autoregressive Machine Translation. [Paper](https://aclanthology.org/2021.naacl-main.313.pdf) & [Code](https://github.com/yongchanghao/multi-task-nat)  
- [20NeurIPS] Incorporating bert into parallel sequence decoding with adapters. [Paper](https://arxiv.org/pdf/2010.06138v1.pdf) & [Code](https://github.com/lemmonation/abnet)
- [21EACL] Non-autoregressive text generation with pre-trained language models. [Paper](https://aclanthology.org/2021.eacl-main.18.pdf) & [Code](https://github.com/yxuansu/NAG-BERT)  
- [22ACL] Universal conditional masked language pre-training for neural machine translation. [Paper](https://arxiv.org/pdf/2203.09210v1.pdf) & [Code](https://github.com/huawei-noah)
- [22EMNLP] Helping the Weak Makes You Strong: Simple Multi-Task Learning Improves Non-Autoregressive Translators. [Paper](https://arxiv.org/pdf/2211.06075.pdf) & [Code](https://arxiv.org/pdf/2211.06075.pdf)


### Others
- [22ICLR] NON-AUTOREGRESSIVE MODELS ARE BETTER MULTILINGUAL TRANSLATORS. [Paper](https://openreview.net/pdf?id=5HvpvYd68b)  
- [21NeurIPS] Duplex Sequence-to-Sequence Learning for Reversible Machine Translation. [Paper](https://arxiv.org/pdf/2105.03458.pdf) & [Code](https://github.com/zhengzx-nlp/REDER)  
- [22NAACL] Non-Autoregressive Machine Translation: It’s Not as Fast as it Seems. [Paper](https://openreview.net/pdf?id=1jg0-AcYVo)
- [22ArXiv] Non-Autoregressive Neural Machine Translation: A Call for Clarity. [Paper](https://arxiv.org/pdf/2205.10577.pdf)
- [UnderReview] ATTENTIVE MLP FOR NON-AUTOREGRESSIVE GENERATION. [Paper](https://openreview.net/pdf?id=hA7XDfCD1y2)

### Results
We show the performance on several datesets without rescoring reported from original paper. * indicates training with knowledge distillation from a big Transformer; ^ denotes training without sequence-level knowledge distillation; # refers to results on IWSLT'16 dataset. 
 
| Model | Iteration |  Speedup | WMT'14 EN-DE | WMT'14 DE-EN | WMT'16  EN-RO | WMT'16 RO-EN | IWSLT'14/16 EN-DE | IWSLT'14/16 DE-EN |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
[FT-NAT](https://arxiv.org/pdf/1711.02281.pdf)| 1 | 15.6x | 17.69 | 21.47  | 27.29 | 29.06  | 26.52# | - |
[RefineNAT](https://aclanthology.org/D18-1149.pdf)| 10 | 1.5x | 21.61  | 25.48 |  29.32 | 30.19 | 27.11# |32.31# |
[RDP](https://arxiv.org/pdf/2012.14583v2.pdf)| 2.5 | 3.5x  | 27.8* | - | - | 33.8 | - | - |
[LRF](https://arxiv.org/pdf/2106.00903.pdf)| 2.5 | 3.5x | 28.2* | - | - | 33.8 | - | -|
[SDMRT](https://arxiv.org/pdf/2112.11640v1.pdf)| 10 | - | 27.72* | 31.65* | 33.72 | 33.94 | 27.49 | - |
[MD](https://aclanthology.org/2020.acl-main.171.pdf)| 1 | - | 25.73 | 30.18 | 31.96 | 33.57 | - | - |
[perLDPE]()| Adaptive | - | 26.3 | 29.5 | - | - | - | - |
[Glat](https://aclanthology.org/2021.acl-long.155.pdf)| 1 | 15.3x | 25.21 | 29.84 |31.19 |32.04 |- |29.61^| 
[PMG](https://aclanthology.org/2021.findings-acl.247.pdf)| 2.5 | 3.5x  | 27.8* | - | -  | 33.8* | - | - | 
[latent-Glat](https://arxiv.org/pdf/2204.02030.pdf)| 1  | 11.3x | 26.64 | 29.93 | - | -| -| 32.47    |  
[Insertion Transformer](https://arxiv.org/pdf/1902.03249.pdf)| log_2(N) | - | 27.41 | - | - |- | - |-| 
[Levenshtein](https://arxiv.org/pdf/1905.11006v1.pdf)| Adaptive | 4.0x | 27.27 | - | - | 33.26 |-|-|   
[CMLM](https://aclanthology.org/D19-1633.pdf)| 10 | 1.7x | 27.03* | 30.53* | 33.08 | 33.31 | - | - | 
[SMART](https://arxiv.org/pdf/2001.08785.pdf)| 10 | 1.7x | 27.65* | 31.27* | - | - | - | - | 
[Disco](https://arxiv.org/pdf/2001.05136.pdf)| Adaptive | 3.5x | 27.34* | 31.31* | 33.22 | 33.25 | - | -|  
[JM-NAT](https://aclanthology.org/2020.acl-main.36.pdf) | 10 | 5.7x | 27.69* | 32.24 * | 33.52 | 33.72 |- |32.59 |     
[Transformer(12-1)](https://arxiv.org/pdf/2006.10369.pdf)| N | 2.5x | 28.3* | 31.8* | 33.8 | 34.8 | - | -| 
[MvSR-NAT](https://arxiv.org/pdf/2108.08447.pdf)| 10 | 3.8x | 27.39* |31.18* |33.38 |33.56  | - |32.55 | 
[Rewrite-NAT](https://aclanthology.org/2021.emnlp-main.265.pdf) | 2.3 | 3.9x | 27.83* | 31.52* | 33.63 |34.09 |- | -|  
[CMLMC](https://openreview.net/pdf?id=I2Hw58KHp8O)| 10 | - | 28.37* | 31.41* | 34.57 |34.13 | 28.51 | 34.78| 
[FlowSeq](https://arxiv.org/pdf/1909.02480v1.pdf)| 1  | 1.1x | 23.72 | 28.39 | 29.73 | 30.72 |27.55 |-|    
[NART-DCRF](https://arxiv.org/pdf/1910.11555.pdf)| 1 | 10.4x | 23.44 | 27.22 | - | - | - | 27.44|   
[PNAT](https://arxiv.org/pdf/1911.10677.pdf)| 1 | 7.3x | 23.05 | 27.18 | - | - | - | 31.23# |   
[SynST](https://aclanthology.org/P19-1122.pdf) | N/6 | 4.6x | 20.74 | 25.50 | -| - | 23.82 | -|  
[LaNAT](https://arxiv.org/pdf/1908.07181v1.pdf)| 1 | 6.8x | 25.10 | - | - | - | - | - |  
[Imputer](https://aclanthology.org/2020.emnlp-main.83.pdf) | 8 | 3.9x | 28.2*  | 31.8* | 34.4 | 34.1 | - | -|    
[LAT](https://arxiv.org/pdf/2011.06132.pdf)| 4 | 6.7x | 27.35 | 32.04 |32.87 | 33.26 |- | 34.08|     
[AligNART](https://aclanthology.org/2021.emnlp-main.1.pdf) | 1 |13.2x | 26.4 | 30.4 | 32.5 | 33.1  | - |- |   
[Reorder-NAT](https://arxiv.org/pdf/1911.02215.pdf)| 1 | 6.0× | 22.79 | 27.28 | 29.30 | 29.50 | 25.29# | -|     
[CNAT](https://aclanthology.org/2021.naacl-main.458.pdf)| 1 | 10.4x | 25.56* | 29.36* | - | - | - | 31.15|    
[SNAT](https://aclanthology.org/2021.eacl-main.105.pdf)| 1 | 22.6x |24.64*  | 28.42* |  32.87 | 32.21 | -| -|    
[Fully-NAT](https://aclanthology.org/2021.findings-acl.11.pdf)| 1 | 16.5x | 27.49 | 31.39 | 33.79 | 34.16 | -| -|   
[ENAT](https://arxiv.org/pdf/1812.09664.pdf)| 1 | 25.3x | 20.65 | 23.02 | 30.08 | - | - | 24.13|     
[NAT-REG](https://arxiv.org/pdf/1902.10245.pdf) | 1 | 27.6x | 20.65 | 24.77 | - | - | 23.14# | 23.89 |     
[LAVA NAT](https://arxiv.org/pdf/2002.03084v1.pdf)| 1 | 20.2x | 27.94 | 31.33 | - | 32.85 | - | 33.59^ |     
[CCAN](https://aclanthology.org/2020.coling-main.389.pdf) | 10 | - | 27.5* | - | - | 33.7 | - | -|   
[DSLP](https://arxiv.org/pdf/2110.07515.pdf)| 1 | 14.8x | 27.02 | 31.61 | 34.17 | 34.60 | - | - |  
[DAD](https://arxiv.org/pdf/2203.16266.pdf) | 1 | 14.7× | 27.51  | 31.96 | 34.68 | 34.98 |- | - |   
[CTC](https://aclanthology.org/2020.emnlp-main.83.pdf)| 1 | 18.6× | 25.7  | 28.10| 32.20 | 31.60 |- | -| 
[RSI-NAT](https://aclanthology.org/P19-1288.pdf) | 1 | 3.6x | 22.27 | 27.25  | 30.57 | 30.83 | 27.78#| -|  
[BoN-Joint](https://arxiv.org/pdf/1911.09320.pdf)| 1 | 9.6x | 20.90 | 24.61 | 28.31 | 29.29 | 25.72# | -|   
[AXE-NAT](https://arxiv.org/pdf/2004.01655.pdf)| 1 | 15.3x | 23.53* | 27.90* | 30.75 | 31.54 | - | -| 
[OAXE-NAT](https://arxiv.org/pdf/2106.05093.pdf) | 1 | 15.3x | 26.10* | 30.20* | 32.40 | 33.30 | - | - |  
[Semi-NAT](https://aclanthology.org/D18-1044.pdf)| N/2 | 1.5x  | 26.90 | - | - | - | - | -  |  
[RecoverSAT](https://aclanthology.org/2020.acl-main.277.pdf)| N/2 | 2.1x | 27.11  | 31.67 | 32.92 | 33.19 | 30.78# | - | 
[Unify](https://aclanthology.org/2020.coling-main.25.pdf)| 10 | - |26.24 | - | - | - | - | 30.73 |  
[GAD](https://arxiv.org/pdf/2203.16487v1.pdf) | 1.6 | 14.3× | 26.48 | - | - | - | - | - |  
[Diformer](https://arxiv.org/pdf/2112.11632v2.pdf) | 10 | - | 27.99 | 31.68 | 34.37 | 33.34 | - | - |  
[Imitate-NAT](https://aclanthology.org/P19-1125.pdf)| 1 | 18.4x | 22.44* | 25.67* | 28.61 |28.90 | 28.41# |-  |  
[Hint-NART](https://aclanthology.org/D19-1573.pdf)| 1 | 30.2x | 21.11  | 25.24 | - | - | - | 25.55 |   
[ENGINE](https://aclanthology.org/2020.acl-main.251.pdf) | 10 | - | - | - | - | 34.04 | - | 33.17 | 
[EM+ODD](https://arxiv.org/pdf/2006.16378.pdf)| 1 | 16.4x | 24.54  | 27.93 | - | - | - | 30.69 |   
[FCL-NAT](http://staff.ustc.edu.cn//linlixu/papers/aaai20a.pdf)| 1 | 28.9x | 21.70 | 25.32 | - | - | - | 26.62 |  
[MULTI-TASK NAT](https://aclanthology.org/2021.naacl-main.313.pdf) | 10 | - | 27.98* | 31.27* | 33.80 | 33.60 | - | - |  
[TCT-NAT](https://www.ijcai.org/Proceedings/2020/0534.pdf)| 1 | 27.6x | 21.94 | 25.62 | - | - | 26.01# | 28.16 | 
[AB-Net](https://arxiv.org/pdf/2010.06138v1.pdf)| - | 2.4x | 28.69* | 33.57* | - | 35.63 | - | 36.49|  
[Bert+CRF-NAT](https://aclanthology.org/2021.eacl-main.18.pdf) | 1 | 11.3x | - | - | - | - | - | 30.45 |  
[CeMAT](https://arxiv.org/pdf/2203.09210v1.pdf)|10 | - | 27.2 | 29.9 | 33.36^ | 33.0^ | 26.7^ | 33.7^ | 


## Speech related (Text to speech, speech translation, automatic speech recognition)
### Papers
### Automatic speech recognition(ASR)
- [22ICASSP] Improving non-autoregressive end-to-end speech recognition with pre-trained acoustic and language models. [Paper](https://arxiv.org/pdf/2201.10103v2.pdf)
- [22ICASSP] Non-Autoregressive ASR with Self-Conditioned Folded Encoders. [Paper](https://arxiv.org/pdf/2202.08474v1.pdf)
- [21ICASSP] CASS-NAT: CTC Alignment-Based Single Step Non-Autoregressive Transformer for Speech Recognition. [Paper](https://arxiv.org/pdf/2010.14725v2.pdf)
- [21ICASSP] Improved Mask-CTC for Non-Autoregressive End-to-End ASR. [Paper](https://arxiv.org/pdf/2010.13270.pdf?ref=https://githubhelp.com)
- [21ICASSP] Non-Autoregressive Transformer ASR with CTC-Enhanced Decoder Input. [Paper](https://arxiv.org/pdf/2010.15025)
- [21NAACL] Align-Refine: Non-Autoregressive Speech Recognition via Iterative Realignment. [Paper](https://aclanthology.org/2021.naacl-main.154.pdf)
- [21arvix] Fast End-to-End Speech Recognition via a Non-Autoregressive Model and Cross-Modal Knowledge Transferring from BERT. [Paper](https://arxiv.org/pdf/2102.07594)
- [21ASRU][ Non-autoregressive Mandarin-English Code-switching Speech Recognition with Pinyin Mask-CTC and Word Embedding Regularization. [Paper](https://arxiv.org/pdf/2104.02258)
- [21ArXiv] Pushing the Limits of Non-Autoregressive Speech Recognition. [Paper](https://arxiv.org/pdf/2104.03416v4.pdf)
- [21ArXiv] WNARS: WFST based Non-autoregressive Streaming End-to-End Speech Recognition. [Paper](https://arxiv.org/pdf/2104.03587v2.pdf)
- [21Interspeech] An Improved Single Step Non-autoregressive Transformer for Automatic Speech Recognition. [Paper](https://arxiv.org/pdf/2106.09885v2.pdf)
- [21ArXiv] Non-autoregressive Transformer with Unified Bidirectional Decoder for Automatic Speech Recognition. [Paper](https://arxiv.org/pdf/2109.06684v1.pdf)
- [21ASRU] A Comparative Study on Non-Autoregressive Modelings for Speech-to-Text Generation. [Paper](https://arxiv.org/pdf/2110.05249v1.pdf)
- [21ArXiv] Boundary and Context Aware Training for CIF-based Non-Autoregressive End-to-end ASR. [Paper](https://arxiv.org/pdf/2104.04702)
- [21ArXiv] Non-autoregressive Transformer-based End-to-end ASR using BERT. [Paper](https://arxiv.org/pdf/2104.04805v1.pdf)
- [21Interspeech ] Multi-Speaker ASR Combining Non-Autoregressive Conformer CTC and Conditional Speaker Chain. [Paper](https://arxiv.org/pdf/2106.08595v1.pdf) & [Code](https://github.com/pengchengguo/espnet)
- [21Interspeech] Streaming End-to-End ASR based on Blockwise Non-Autoregressive Models. [Paper](https://arxiv.org/pdf/2107.09428v1.pdf) & [Code](https://github.com/espnet/espnet)
- [21ArXiv] Listen and Fill in the Missing Letters: Non-Autoregressive Transformer for Speech Recognition. [Paper](https://arxiv.org/pdf/1911.04908.pdf)
- [21INTERSPEECH] Mask CTC: Non-Autoregressive End-to-End ASR with CTC and Mask Predict. [Paper](https://arxiv.org/pdf/2005.08700.pdf) & [Code](https://github.com/espnet/espnet)
- [21INTERSPEECH] Insertion-Based Modeling for End-to-End Automatic Speech Recognition. [Paper](https://arxiv.org/pdf/2005.13211.pdf) & [Code](https://github.com/espnet/espnet)
- [21INTERSPEECH] Listen Attentively, and Spell Once: Whole Sentence Generation via a Non-Autoregressive Architecture for Low-Latency Speech Recognition. [Paper](https://arxiv.org/pdf/2005.04862v4.pdf)
- [21ArXiv] Spike-Triggered Non-Autoregressive Transformer for End-to-End Speech Recognition. [Paper](https://arxiv.org/pdf/2005.07903v1.pdf)
- [21IAASP] INTERMEDIATE LOSS REGULARIZATION FOR CTC-BASED SPEECH RECOGNITION. [Paper](https://arxiv.org/pdf/2102.03216v1.pdf) & [Code](https://arxiv.org/pdf/2102.03216v1.pdf)
- [21INTERSPEECH] Align-Denoise: Single-Pass Non-Autoregressive Speech Recognition. [Paper](http://dx.doi.org/10.21437/Interspeech.2021-1906) & [Code](https://github.com/bobchennan/espnet/tree/align_denoise)
- [21ICCASP] Intermediate loss regularization for ctc-based speech recognition. [Paper](https://arxiv.org/pdf/2102.03216v1.pdf) & [Code](https://github.com/espnet/espnet) 
- [21ArXiv] Relaxing the conditional independence assumption of CTC-based ASR by conditioning on intermediate predictions. [Paper](https://arxiv.org/pdf/2104.02724.pdf) & [Code](https://github.com/espnet/espnet)
- [22ArXiv] Improving CTC-based ASR Models with Gated Interlayer Collaboration. [Paper](https://arxiv.org/pdf/2205.12462.pdf)
- [UnderReview] PATCORRECT: NON-AUTOREGRESSIVE PHONEMEAUGMENTED TRANSFORMER FOR ASR ERROR CORRECTION. [Paper](https://openreview.net/pdf?id=njAes-sX0m)
- [22SLT] A context-aware knowledge transferring strategy for CTC-based ASR. [Paper](https://arxiv.org/pdf/2210.06244.pdf)
- [22Interspeech] Non-autoregressive Error Correction for CTC-based ASR with Phone-conditioned Masked LM. [Paper](https://arxiv.org/pdf/2209.04062.pdf)
### Text to speech (TTS)
- [22INTERSPEECH] Hierarchical and Multi-Scale Variational Autoencoder for Diverse and Natural Non-Autoregressive Text-to-Speech. [Paper](https://arxiv.org/pdf/2204.04004.pdf)
- [22ArXiv] vTTS: visual-text to speech. [Paper](https://arxiv.org/pdf/2203.14725.pdf)
- [22Interspeech] A Multi-Scale Time-Frequency Spectrogram Discriminator for GAN-based Non-Autoregressive TTS. [Paper](https://arxiv.org/pdf/2203.01080.pdf)
- [22ACL] Revisiting Over-Smoothness in Text to Speech. [Paper](https://arxiv.org/pdf/2202.13066.pdf)
- [21ICLR] Bidirectional Variational Inference for Non-Autoregressive Text-to-Speech. [Paper](https://openreview.net/pdf?id=o3iritJHLfO) & [Code](https://github.com/LEEYOONHYUNG/BVAE-TTS
- [21ArXiv] VARA-TTS: Non-Autoregressive Text-to-Speech Synthesis based on Very Deep VAE with Residual Attention. [Paper](https://arxiv.org/pdf/2102.06431v1.pdf)
- [21ArXiv] Nana-HDR: A Non-attentive Non-autoregressive Hybrid Model for TTS. [Paper](https://arxiv.org/pdf/2109.13673.pdf)
- [21Speech Synthesis Workshop] Non-Autoregressive TTS with Explicit Duration Modelling for Low-Resource Highly Expressive Speech. [Paper](https://arxiv.org/pdf/2106.12896v2.pdf)
- [21ArXiv] VAENAR-TTS: Variational Auto-Encoder based Non-AutoRegressive Text-to-Speech Synthesis. [Paper](https://arxiv.org/pdf/2107.03298v1.pdf) & [Code](https://github.com/thuhcsi/VAENAR-TTS)
- [21ArXiv] Exploring Timbre Disentanglement in Non-Autoregressive Cross-Lingual Text-to-Speech. [Paper](https://arxiv.org/pdf/2110.07192v1.pdf) 
- [21ICML] Non-Autoregressive Neural Text-to-Speech [Paper](https://arxiv.org/pdf/1905.08459.pdf) & [Code](https://github.com/ksw0306/WaveVAE)
- [21Interspeech] Quasi-Periodic Parallel WaveGAN Vocoder: A Non-autoregressive Pitch-dependent Dilated Convolution Model for Parametric Speech Generation. [Paper](https://arxiv.org/pdf/2005.08654v1.pdf) & [Code](https://github.com/bigpon/QPPWG)
- [21NeurIPS] FastSpeech: Fast, Robust and Controllable Text to Speech. [Paper](https://arxiv.org/pdf/1905.09263.pdf) & [Code](https://github.com/coqui-ai/TTS)
- [21ArXiv] TalkNet 2: Non-Autoregressive Depth-Wise Separable Convolutional Model for Speech Synthesis with Explicit Pitch and Duration Prediction. [Paper](https://arxiv.org/pdf/2104.08189v3.pdf) & [Code](https://github.com/rishikksh20/TalkNet2-pytorch))
- [21ArXiv] Hierarchical Prosody Modeling for Non-Autoregressive Speech Synthesis. [Paper](https://arxiv.org/pdf/2011.06465v3.pdf) & [Code](https://github.com/ming024/FastSpeech2)
- [22ArXiv] Controllable and Lossless Non-Autoregressive End-to-End Text-to-Speech. [Paper](https://arxiv.org/pdf/2207.06088.pdf) & 
- [22ACL] Cross-Utterance Conditioned VAE for Non-Autoregressive Text-to-Speech. [Paper](https://arxiv.org/pdf/2205.04120.pdf)
- [UnderReview] BAG OF TRICKS FOR UNSUPERVISED TTS. [Paper](https://openreview.net/pdf?id=SbR9mpTuBn).
- [23ICASSP] Spoofed training data for speech spoofing countermeasure can be efficiently created using neural vocoders. [Paper](https://arxiv.org/pdf/2210.10570.pdf)

### Speech translation 
- [21ACL findings] Investigating the Reordering Capability in CTC-based Non-Autoregressive End-to-End Speech Translation. [Paper](https://arxiv.org/pdf/2105.04840v1.pdf) & [Code](https://github.com/voidism/NAR-ST)
- [21ICASSP] ORTHROS: non-autoregressive end-to-end speech translation With dual-decoder. [Paper](https://arxiv.org/pdf/2010.13047))
- [21ArXiv] Non-autoregressive End-to-end Speech Translation with Parallel Autoregressive Rescoring. [Paper](https://arxiv.org/pdf/2109.04411v1.pdf)
- [21ASRU] Fast-MD: Fast Multi-Decoder End-to-End Speech Translation with Non-Autoregressive Hidden Intermediates. [Paper](https://arxiv.org/pdf/2109.12804v1.pdf)
- [22ArXiv] Controllable and Lossless Non-Autoregressive End-to-End Text-to-Speech. [Paper](https://arxiv.org/pdf/2207.06088.pdf)
- [22ArXiv] A Novel Chinese Dialect TTS Frontend with Non-Autoregressive Neural Machine Translation. [Paper](https://arxiv.org/pdf/2206.04922.pdf)
- [UnderReview] TranSpeech: Speech-to-Speech Translation With Bilateral Perturbation. [Paper](https://openreview.net/pdf?id=UVAmFAtC5ye)  
### Others
- [22ArXiv] Conditional Deep Hierarchical Variational Autoencoder for Voice Conversion. [Paper](https://arxiv.org/pdf/2112.02796.pdf)
- [22ICLR] Generating Videos with Dynamics-aware Implicit Generative Adversarial Networks. [Paper](https://arxiv.org/pdf/2202.10571.pdf)
- [21ICASSP] Non-Autoregressive Sequence-To-Sequence Voice Conversion. [Paper](https://arxiv.org/pdf/2104.06793)
- [21ArXiv] Non-Autoregressive Predictive Coding for Learning Speech Representations from Local Dependencies. [Paper](https://arxiv.org/pdf/2011.00406v1.pdf) & [Code](https://github.com/Alexander-H-Liu/NPC)
- [21ArXiv] Exploring Non-Autoregressive End-To-End Neural Modeling For English Mispronunciation Detection And Diagnosis. [Paper](https://arxiv.org/ftp/arxiv/papers/2111/2111.00844.pdf)
- [22ArXiv] FastLTS: Non-Autoregressive End-to-End Unconstrained Lip-to-Speech Synthesis. [Paper](https://arxiv.org/pdf/2207.03800.pdf)
- [22ArXiv] Streaming non-autoregressive model for any-to-many voice conversion. [Paper](https://arxiv.org/pdf/2206.07288.pdf)
- [UnderReview] REPHRASETTS: DYNAMIC LENGTH TEXT BASED SPEECH INSERTION WITH SPEAKER STYLE TRANSFER. [Paper](https://openreview.net/pdf?id=8zsK9lbna9L)
- [22ArXiv] Personalization of CTC Speech Recognition Models. [Paper](https://arxiv.org/pdf/2210.09510.pdf)
- [22Interspeech] Knowledge Transfer and Distillation from Autoregressive to Non-Autoregressive Speech Recognition. [Paper](https://arxiv.org/pdf/2207.10600.pdf)

## Other tasks (Text Summarization; Dialogue and Intent Detection; Grammatical Error Correction; Text Style Transfer; Parsing; etc.)
### Papers
### General-Purpose methods
- [22ArXiv] An Imitation Learning Curriculum for Text Editing with Non-Autoregressive Models. [Paper](https://arxiv.org/pdf/2203.09486.pdf)
- [21ACL-IJCNLP] POS-Constrained Parallel Decoding for Non-autoregressive Generation. [Paper](https://aclanthology.org/2021.acl-long.467.pdf) & [Code](https://github.com/yangkexin/pospd)
- [21ArXiv] Integrated Training for Sequence-to-Sequence Models Using Non-Autoregressive Transformer. [Paper](https://arxiv.org/pdf/2109.12950v1.pdf)
- [21ArXiv] Improving Non-autoregressive Generation with Mixup Training. [Paper](https://arxiv.org/pdf/2110.11115v1.pdf) & [Code](https://github.com/kongds/mist)
- [21EACL] Non-Autoregressive Text Generation with Pre-trained Language Models. [Paper](https://arxiv.org/pdf/2102.08220v1.pdf) & [Code](https://github.com/yxuansu/NAG-BERT)
- [21ICML] BANG: Bridging Autoregressive and Non-autoregressive Generation with Large Scale Pretraining. [Paper](https://arxiv.org/pdf/2012.15525v3.pdf) & [Code](https://github.com/microsoft/BANG)
- [22ArXiv] EdiT5: Semi-Autoregressive Text-Editing with T5 Warm-Start. [Paper](https://arxiv.org/pdf/2205.12209.pdf)
- [22ArXiv] A Self-Paced Mixed Distillation Method for Non-Autoregressive Generation. [Paper](https://arxiv.org/pdf/2205.11162.pdf)
- [22EMNLP] ELMER: A Non-Autoregressive Pre-trained Language Model for Efficient and Effective Text Generation. [Paper](https://arxiv.org/pdf/2210.13304.pdf) &   [Code](https://github.com/RUCAIBox/ELMER)
### Summarization
- [22ACL] Learning Non-Autoregressive Models from Search for Unsupervised Sentence Summarization. [Paper](https://openreview.net/pdf?id=UNzc8gReN7m).
- [22Arxiv] A Character-Level Length-Control Algorithm for Non-Autoregressive Sentence Summarization. [Paper](https://arxiv.org/pdf/2205.14522.pdf).
### Dialogue
- [21EMNLP] Thinking Clearly, Talking Fast: Concept-Guided Non-Autoregressive Generation for Open-Domain Dialogue Systems. [Paper](https://arxiv.org/pdf/2109.04084v1.pdf) & [Code](https://github.com/rowitzou/cg-nar)
- [20ArXiv] Non-Autoregressive Neural Dialogue Generation. [Paper](https://arxiv.org/pdf/2002.04250v2.pdf)
- [21ACL-IJCNLP] GL-GIN: Fast and Accurate Non-Autoregressive Model for Joint Multiple Intent Detection and Slot Filling. [Paper](https://arxiv.org/pdf/2106.01925v1.pdf) & [Code](https://github.com/yizhen20133868/GL-GIN)
- [21ArXiv] An Effective Non-Autoregressive Model for Spoken Language Understanding. [Paper](https://arxiv.org/pdf/2108.07005v1.pdf)
- [20EMNLP] SlotRefine: A Fast Non-Autoregressive Model for Joint Intent Detection and Slot Filling. [Paper](https://arxiv.org/pdf/2010.02693v2.pdf) & [Code](https://github.com/moore3930/SlotRefine)
- [20ICLR] Non-Autoregressive Dialog State Tracking. [Paper](https://arxiv.org/pdf/2002.08024v1.pdf) & [Code](https://github.com/henryhungle/NADST)

### Semantic parsing
- [21EMNLP Findings] Span Pointer Networks for Non-Autoregressive Task-Oriented Semantic Parsing. [Paper](https://arxiv.org/pdf/2104.07275v3.pdf)
- [21NAACL-HLT] Non-Autoregressive Semantic Parsing for Compositional Task-Oriented Dialog. [Paper](https://arxiv.org/pdf/2104.04923v1.pdf) & [Code](https://github.com/facebookresearch/pytext)
- [20ArXiv] Recursive Non-Autoregressive Graph-to-Graph Transformer for Dependency Parsing with Iterative Refinement. [Paper](https://arxiv.org/pdf/2003.13118v2.pdf) & [Code](https://github.com/idiap/g2g-transformer)

### Grammatical Error Correction
- [21ACL-IJCNLP] Tail-to-Tail Non-Autoregressive Sequence Prediction for Chinese Grammatical Error Correction. [Paper](https://arxiv.org/pdf/2106.01609v3.pdf) & [Code](https://github.com/lipiji/TtT)
- [21WNUT] Character Transformations for Non-Autoregressive GEC Tagging. [Paper](https://arxiv.org/pdf/2111.09280v1.pdf) & [Code](https://github.com/ufal/wnut2021_character_transformations_gec)

### Others
- [21ACL-IJCNLP Findings] A Non-Autoregressive Edit-Based Approach to Controllable Text Simplification. [Paper](https://aclanthology.org/2021.findings-acl.330.pdf)
- [21ACL-IJCNLP Findings] NAST: A Non-Autoregressive Generator with Word Alignment for Unsupervised Text Style Transfer. [Paper](https://arxiv.org/pdf/2106.02210v1.pdf) & [Code](https://github.com/thu-coai/NAST)
- [21EMNLP] Exploring Non-Autoregressive Text Style Transfer. [Paper](https://aclanthology.org/2021.emnlp-main.730.pdf) & [Code](https://github.com/sunlight-ym/nar_style_transfer)
- [21ArXiv] EncT5: Fine-tuning T5 Encoder for Non-autoregressive Tasks. [Paper](https://arxiv.org/pdf/2110.08426v1.pdf)
- [21EMNLP] Maximal Clique Based Non-Autoregressive Open Information Extraction. [Paper](https://aclanthology.org/2021.emnlp-main.764.pdf)
- [20ArXiv] A Study on the Autoregressive and non-Autoregressive Multi-label Learning. [Paper](https://arxiv.org/pdf/2012.01711v1.pdf)
- [22ArXiv]KECP: Knowledge Enhanced Contrastive Prompting for Few-shot Extractive Question Answering. [Paper](https://arxiv.org/pdf/2205.03071.pdf) & [Code](https://github.com/alibaba/EasyNLP)
- [22ArXiv] Diffusion-LM Improves Controllable Text Generation. [Paper](https://arxiv.org/pdf/2205.14217.pdf) & [Code](https://github.com/XiangLi1999/Diffusion-LM)
- [22ArXiv] Capture Salient Historical Information: A Fast and Accurate Non-Autoregressive Model for Multi-turn Spoken Language Understanding. [Paper](https://arxiv.org/pdf/2206.12209.pdf)
- [UnderReview] NAPG: NON-AUTOREGRESSIVE PROGRAM GENERATION FOR HYBRID TABULAR TEXTUAL QUESTION ANSWERING. [Paper](https://openreview.net/pdf?id=Q31C6XQOEvl)
- [22ARXIV] Bilingual Synchronization: Restoring Translational Relationships with Editing Operations. [Paper](https://arxiv.org/pdf/2210.13163.pdf)
- [22ArXiv] Acoustic-aware Non-autoregressive Spell Correction with Mask Sample Decoding. [Paper](https://arxiv.org/pdf/2210.08665.pdf)
- [22ArXiv] Continuous conditional video synthesis by neural processes. [Paper](https://arxiv.org/pdf/2210.05810.pdf) 


## Computer Vision
- [UnderReview] Semi-Autoregressive Energy Flows: Towards Determinant-Free Training of Normalizing Flows. [Paper](https://openreview.net/forum?id=GBU1mm8_WkV)
- [UnderReview] LANGUAGE-GUIDED ARTISTIC STYLE TRANSFER USING THE LATENT SPACE OFDALL-E. [Paper](https://openreview.net/pdf?id=yDx3GP7Qjfl)
- [UnderReview] CHIRODIFF: MODELLING CHIROGRAPHIC DATA WITH DIFFUSION MODELS. [Paper](https://openreview.net/pdf?id=1ROAstc9jv)
- [22NeurIPS] Learning Distinct and Representative Modes for Image Captioning. [Paper](https://arxiv.org/pdf/2209.08231.pdf) & [Code](https://github.com/bladewaltz1/ModeCap)
- [22ArXiv] STPOTR: Simultaneous Human Trajectory and Pose Prediction Using a Non-Autoregressive Transformer for Robot Following Ahead. [Paper](https://arxiv.org/pdf/2209.07600.pdf)
- [22ECCV] Explicit Image Caption Editing. [Paper](https://arxiv.org/pdf/2207.09625.pdf) & [Code](https://github.com/baaaad/ECE)
- [22ECCV] Improved Masked Image Generation with Token-Critic. [Paper](https://arxiv.org/pdf/2209.04439.pdf)
- [223DV] TEACH: Temporal Action Composition for 3D Humans. [Paper](https://arxiv.org/pdf/2209.04066.pdf)
- [23ECCV] Non-Autoregressive Sign Language Production via Knowledge Distillation. [Paper](https://arxiv.org/pdf/2208.06183.pdf)

## Specially, we present recent progress of difussion models in different tasks, which also adpot non-autoregressive format in each difffusion step.
## Difussion Models
- [21ICLR] WAVEGRAD: ESTIMATING GRADIENTS FOR WAVEFORM GENERATION. [Paper](https://arxiv.org/pdf/2009.00713.pdf)
- [21NeuIPS] Structured Denoising Diffusion Models in Discrete State-Spaces. [Paper](https://arxiv.org/pdf/2210.16886.pdf)
- [22ArXiv] Photorealistic text-to-image diffusion models with deep language understanding. [Paper](https://arxiv.org/pdf/2205.11487.pdf)
- [22ArXiv] Hierarchical Text-Conditional Image Generation with CLIP Latents. [Paper](https://arxiv.org/pdf/2204.06125.pdf)
- [22ICML] Glide: Towards photorealistic image generation and editing with text-guided diffusion models. [Paper](https://arxiv.org/pdf/2112.10741.pdf)
- [22ArXiv] Classifier-free diffusion guidance. [Paper](https://arxiv.org/pdf/2207.12598.pdf)
- [UnderReview] CHIRODIFF: MODELLING CHIROGRAPHIC DATA WITH DIFFUSION MODELS. [Paper](https://openreview.net/pdf?id=1ROAstc9jv)
- [22ICML]  Latent diffusion energy based model for interpretable text modeling. [Paper](https://arxiv.org/abs/2206.05895) 
- [22ArXiv] Diffusion-lm improves controllable text generation. [Paper](https://arxiv.org/pdf/2205.14217.pdf) & [Code](https://github.com/XiangLi1999/Diffusion-LM.git)
- [UnderReview] DIFFUSER: DIFFUSION VIA EDIT-BASED RECONSTRUCTION. [Paper](https://arxiv.org/pdf/2210.16886.pdf) & [Code](https://github.com/machelreid/diffuser)
- [22ArXiv] DIFFUSEQ: SEQUENCE TO SEQUENCE TEXT GENERATION WITH DIFFUSION MODELS. [Paper](https://arxiv.org/pdf/2210.08933.pdf) & [Code](https://github.com/Shark-NLP/DiffuSeq)
- [22ArXiv] Understanding Diffusion Models: A Unified Perspective. [Paper](https://arxiv.org/pdf/2208.11970.pdf)
- [22ArXiv] Diffusion Models: A Comprehensive Survey of Methods and Applications. [Paper](https://arxiv.org/pdf/2209.00796.pdf)
