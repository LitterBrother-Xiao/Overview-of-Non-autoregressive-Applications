# Overview-of-Non-autoregressive-Applications
This repo presents an overview of Non-autoregressive (NAR) models, including links to related papers and corresponding codes.

NAR models aim to speed up decoding and reduce the inference latency, then realize better industry application. However, this improvement of speed comes at the expense of the decline of quality. Many methods and tricks are proposed to reduce this gap.

NAR models are first proposed for neural machine translation, and then are applied for various tasks, such as speech to text, speech gneration, speech translation, text summarization; dialogue and intent detection; grammatical error correction; text style transfer; semantic parsing and etc.

A survey on non-autoregressive neural machine translation including a brief review of other various tasks can be found here. 

## Neural machine translation
### Papers
### Data manipulation
##### &nbsp;&nbsp;&nbsp;&nbsp; Knowledge distillation
- [20ICLR] UNDERSTANDING KNOWLEDGE DISTILLATION IN NON-AUTOREGRESSIVE MACHINE TRANSLATION. [Paper](https://arxiv.org/pdf/1911.02727.pdf.) &
[Code](https://github.com/pytorch/fairseq/tree/master/examples/nonautoregressive_translation.)  
- [20ACL] A Study of Non-autoregressive Model for Sequence Generation. [Paper](https://aclanthology.org/2020.acl-main.15.pdf)  
- [20ACL] Improving Non-autoregressive Neural Machine Translation with Monolingual Data. [Ppaer](https://aclanthology.org/2020.acl-main.171.pdf) 
- [21ICLR] UNDERSTANDING AND IMPROVING LEXICAL CHOICE IN NON-AUTOREGRESSIVE TRANSLATION. [Paper](https://arxiv.org/pdf/2012.14583v2.pdf)  
- [21ACL-IJCNLP] Rejuvenating Low-Frequency Words: Making the Most of Parallel Data in Non-Autoregressive Translation. [Paper](https://arxiv.org/pdf/2106.00903.pdf) & [Code](https://github.com/longyuewangdcu/RLFW-NAT)  
- [22Findings of ACL-IJCNLP] How Does Distilled Data Complexity Impact the Quality and Confidence of Non-Autoregressive Machine Translation? [Paper](https://aclanthology.org/2021.findings-acl.385.pdf)
- [UnderReview] Self-Distillation Mixup Training for Non-autoregressive Neural Machine Translation. [Paper](https://arxiv.org/pdf/2112.11640v1.pdf)  
- [UnderReview] Neighbors Are Not Strangers: Improving Non-Autoregressive Translation under Low-Frequency Lexical Constraints. [Paper](https://openreview.net/pdf?id=T-Wh9Ds-qk)

##### &nbsp;&nbsp;&nbsp;&nbsp; Data learning strategy 
- [21ACL-IJCNLP] Glancing Transformer for Non-Autoregressive Neural Machine Translation. [Paper](https://aclanthology.org/2021.acl-long.155.pdf) & [Code](https://github.com/FLC777/GLAT)  
- [21ACL-IJCNLP Findinds] Progressive Multi-Granularity Training for Non-Autoregressive Translation. [Paper](https://aclanthology.org/2021.findings-acl.247.pdf) 
- [21ArXiv] MvSR-NAT: Multi-view Subset Regularization for Non-Autoregressive Machine Translation. [Paper](https://arxiv.org/pdf/2108.08447.pdf)  
- [22ACL] latent-GLAT: Glancing at Latent Variables for Parallel Text Generation. [Paper](https://arxiv.org/pdf/2204.02030.pdf) & [Code](https://github.com/baoy-nlp/Latent-GLAT)
- [UnderReview] Non-Autoregressive Neural Machine Translation with Consistency Regularization Optimized Variational Framework. [Paper](https://openreview.net/pdf?id=cLe29FcNAKb)
- [UnderReview] Contrastive Conditional Masked Language Model for Non-autoregressive Neural Machine Translation. [Paper](https://openreview.net/pdf?id=9_j8yJ6ISSr)

### Modeling 
##### &nbsp;&nbsp;&nbsp;&nbsp;Iteration-based methods
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
- [22ICLR] IMPROVING NON-AUTOREGRESSIVE TRANSLATION MODELS WITHOUT DISTILLATION. [Paper](https://openreview.net/pdf?id=I2Hw58KHp8O) 

##### &nbsp;&nbsp;&nbsp;&nbsp;Latent variable-based methods
- [18ICLR] NON-AUTOREGRESSIVE NEURAL MACHINE TRANSLATION. [Paper](https://arxiv.org/pdf/1711.02281.pdf) & [Code](https://github.com/salesforce/nonauto-nmt)
- [19EMNLP-IJCNLP] FlowSeq: Non-Autoregressive Conditional Sequence Generation with Generative Flow. [Paper](https://arxiv.org/pdf/1909.02480v1.pdf) & [Code](https://github.com/XuezheMax/flowseq)
- [19NeurIPS] Fast Structured Decoding for Sequence Models. [Paper](https://arxiv.org/pdf/1910.11555.pdf) 
- [19ArXiv] Non-autoregressive Transformer by Position Learning. [Paper](https://arxiv.org/pdf/1911.10677.pdf)  
- [19ACL] Syntactically Supervised Transformers for Faster Neural Machine Translation. [Paper](https://aclanthology.org/P19-1122.pdf) & [Code](https://github.com/dojoteef/synst)
- [20AAAI] Latent-Variable Non-Autoregressive Neural Machine Translation with Deterministic Inference using a Delta Posterior. [Paper](https://arxiv.org/pdf/1908.07181v1.pdf) & - [Code](https://github.com/zomux/lanmt)   
- [20EMNLP] Non-Autoregressive Machine Translation with Latent Alignments. [Paper](https://aclanthology.org/2020.emnlp-main.83.pdf)  
- [20ArXiv] Incorporating a Local Translation Mechanism into Non-autoregressive Translation. [Paper](https://arxiv.org/pdf/2011.06132.pdf) & [Code](https://github.com/shawnkx/NAT-with-Local-AT)
- [21EMNLP] AligNART: Non-autoregressive Neural Machine Translation by Jointly Learning to Estimate Alignment and Translate. [Paper](https://aclanthology.org/2021.emnlp-main.1.pdf)  
- [21EACL] Enriching Non-Autoregressive Transformer with Syntactic and Semantic Structures for Neural Machine Translation. [Paper](https://aclanthology.org/2021.eacl-main.105.pdf)   
- [21AAAI] Guiding Non-Autoregressive Neural Machine Translation Decoding with Reordering Information. [Paper](https://arxiv.org/pdf/1911.02215.pdf) & [Code](https://github.com/ranqiu92/ReorderNAT)  
- [21NAACL-HLT] Non-Autoregressive Translation by Learning Target Categorical Codes. [Paper](https://aclanthology.org/2021.naacl-main.458.pdf) & [Code](https://github.com/baoy-nlp/CNAT)  
- [21ACL-IJCNLP Findinds] Fully Non-autoregressive Neural Machine Translation:Tricks of the Trade. [Paper](https://aclanthology.org/2021.findings-acl.11.pdf) & [Code](https://github.com/pytorch/fairseq/tree/main/examples/nonautoregressive_translation)  

##### &nbsp;&nbsp;&nbsp;&nbsp;Other enhancements-based mothods
- [19AAAI] Non-Autoregressive Neural Machine Translation with Enhanced Decoder Input. [Paper](https://arxiv.org/pdf/1812.09664.pdf)  
- [19AAAI] Non-Autoregressive Machine Translation with Auxiliary Regularization. [Paper](https://arxiv.org/pdf/1902.10245.pdf)   
- [20COLING] Context-Aware Cross-Attention for Non-Autoregressive Translation. [Paper](https://aclanthology.org/2020.coling-main.389.pdf)  
- [21ArXiv] LAVA NAT: A Non-Autoregressive Translation Model with Look-Around Decoding and Vocabulary Attention. [Paper](https://arxiv.org/pdf/2002.03084v1.pdf)
- [22ArXiv] Non-autoregressive Translation with Dependency-Aware Decoder. [Paper](https://arxiv.org/pdf/2203.16266.pdf) & [Code](https://github.com/zja-nlp/NAT_with_DAD)

### Criterion
- [06ICML] Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks. [Paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf) & [Code](https://github.com/parlance/ctcdecode)  
- [18EMNLP] End-to-End Non-Autoregressive Neural Machine Translation with Connectionist Temporal Classification. [Paper](https://aclanthology.org/D18-1336.pdf)  
- [19ACL] Retrieving Sequential Information for Non-Autoregressive Neural Machine Translation. [Paper](https://aclanthology.org/P19-1288.pdf) & [Code](https://github.com/ictnlp/RSI-NAT)  
- [20AAAI] Minimizing the Bag-of-Ngrams Difference for Non-Autoregressive Neural Machine Translation. [Paper](https://arxiv.org/pdf/1911.09320.pdf) & [Code](https://github.com/ictnlp/BoN-NAT)  
- [20ICML] Aligned Cross Entropy for Non-Autoregressive Machine Translation. [Paper](https://arxiv.org/pdf/2004.01655.pdf) & [Code](https://github.com/m3yrin/aligned-cross-entropy)  
- [21ICML] Order-Agnostic Cross Entropy for Non-Autoregressive Machine Translation. [Paper](https://arxiv.org/pdf/2106.05093.pdf) & [Code](https://github.com/tencent-ailab/ICML21_OAXE)
- [UnderReview] Maximum Proxy-Likelihood Estimation for Non-autoregressive Machine Translation. [Paper](https://openreview.net/pdf?id=ps4ihHcV19)

### Decoding
- [18EMNLP] Semi-Autoregressive Neural Machine Translation. [Paper](https://aclanthology.org/D18-1044.pdf)   
- [20ACL] Learning to Recover from Multi-Modality Errors for Non-Autoregressive Neural Machine Translation. [Paper](https://aclanthology.org/2020.acl-main.277.pdf) & [Code](https://github.com/ranqiu92/RecoverSAT) 
- [20COLING] Train Once, and Decode As You Like. [Paper](https://aclanthology.org/2020.coling-main.25.pdf)
- [UnderReview] Diformer: Directional Transformer for Neural Machine Translation. [Paper](https://arxiv.org/pdf/2112.11632v2.pdf)
- [22ArXiv] Lossless Speedup of Autoregressive Translation with Generalized Aggressive Decoding. [Paper](https://arxiv.org/pdf/2203.16487v2.pdf) & [Code](https://github.com/hemingkx/Generalized-Aggressive-Decoding)

### Benefiting from Pre-trained Modoels
- [19ACL] Imitation Learning for Non-Autoregressive Neural Machine Translation. [Paper](https://aclanthology.org/P19-1125.pdf)  
- [19EMNLP-IJCNLP] Hint-Based Training for Non-Autoregressive Machine Translation. [Paper](https://aclanthology.org/D19-1573.pdf) & [Code](https://github.com/zhuohan123/hint-nart)  
- [20AAAI] Fine-Tuning by Curriculum Learning for Non-Autoregressive Neural Machine Translation. [Paper](http://staff.ustc.edu.cn/~linlixu/papers/aaai20a.pdf) & [Code](https://github.com/lemmonation/fcl-nat)  
- [20ICML] An EM Approach to Non-autoregressive Conditional Sequence Generation. [Paper](https://arxiv.org/pdf/2006.16378.pdf) & [Code](https://github.com/Edward-Sun/NAT-EM)  
- [20ACL] ENGINE: Energy-Based Inference Networks for Non-Autoregressive Machine Translation. [Paper](https://aclanthology.org/2020.acl-main.251.pdf) & [Code]( https://github.com/lifu-tu/ENGINE)
- [20AutoSimtrans] Improving Autoregressive NMT with Non-Autoregressive Model. [Paper](https://aclanthology.org/2020.autosimtrans-1.4.pdf)  
- [21IJCAI] Task-Level Curriculum Learning for Non-Autoregressive Neural Machine Translation. [Ppaer](https://www.ijcai.org/Proceedings/2020/0534.pdf)  
- [21NAACL-HLT] Multi-Task Learning with Shared Encoder for Non-Autoregressive Machine Translation. [Paper](https://aclanthology.org/2021.naacl-main.313.pdf) & [Code](https://github.com/yongchanghao/multi-task-nat)  
-[20NeurIPS] Incorporating bert into parallel sequence decoding with adapters. [Paper](https://arxiv.org/pdf/2010.06138v1.pdf) & [Code](https://github.com/lemmonation/abnet)
-[21EACL] Non-autoregressive text generation with pre-trained language models. [Paper](https://aclanthology.org/2021.eacl-main.18.pdf) & [Code](https://github.com/yxuansu/NAG-BERT)
-[22ArXiv] Universal conditional masked language pre-training for neural machine transl. [Paper](https://arxiv.org/pdf/2203.09210v1.pdf) & [Code](https://github.com/huawei-noah)

### Others
- [22ICLR] NON-AUTOREGRESSIVE MODELS ARE BETTER MULTILINGUAL TRANSLATORS. [Paper](https://openreview.net/pdf?id=5HvpvYd68b)  
- [21NeurIPS] Duplex Sequence-to-Sequence Learning for Reversible Machine Translation. [Paper](https://arxiv.org/pdf/2105.03458.pdf) & [Code](https://github.com/zhengzx-nlp/REDER)  
- [UnderReview] Non-Autoregressive Machine Translation: It’s Not as Fast as it Seems. [Paper](https://openreview.net/pdf?id=1jg0-AcYVo)


## Speech related(Text to speech, speech translation, automatic speech recognition )
# Papers
### 2022
- [ICASSP] Improving non-autoregressive end-to-end speech recognition with pre-trained acoustic and language models. [Paper](https://arxiv.org/pdf/2201.10103v2.pdf)
- [ICASSP] Non-Autoregressive ASR with Self-Conditioned Folded Encoders. [Paper](https://arxiv.org/pdf/2202.08474v1.pdf)
- [ArXiv] vTTS: visual-text to speech. [Paper](https://arxiv.org/pdf/2203.14725.pdf)
- [Interspeech] A Multi-Scale Time-Frequency Spectrogram Discriminator for GAN-based Non-Autoregressive TTS. [Paper](https://arxiv.org/pdf/2203.01080.pdf)
- [ACL] Revisiting Over-Smoothness in Text to Speech. [Paper](https://arxiv.org/pdf/2202.13066.pdf)
- [ICLR]Generating Videos with Dynamics-aware Implicit Generative Adversarial Networks. [Paper](https://arxiv.org/pdf/2202.10571.pdf)
### 2021
- [ACL findings] Investigating the Reordering Capability in CTC-based Non-Autoregressive End-to-End Speech Translation. [Paper](https://arxiv.org/pdf/2105.04840v1.pdf) & [Code](https://github.com/voidism/NAR-ST)
- [ICASSP] CASS-NAT: CTC Alignment-Based Single Step Non-Autoregressive Transformer for Speech Recognition. [Paper](https://arxiv.org/pdf/2010.14725v2.pdf)
- [ICASSP] Non-Autoregressive Sequence-To-Sequence Voice Conversion. [Paper](https://arxiv.org/pdf/2104.06793)
- [ICASSP] Improved Mask-CTC for Non-Autoregressive End-to-End ASR. [Paper](https://arxiv.org/pdf/2010.13270.pdf?ref=https://githubhelp.com)
- [ICASSP] ORTHROS: non-autoregressive end-to-end speech translation With dual-decoder. [Paper](https://arxiv.org/pdf/2010.13047)
- [ICASSP] Non-Autoregressive Transformer ASR with CTC-Enhanced Decoder Input. [Paper](https://arxiv.org/pdf/2010.15025)
- [ICLR] Bidirectional Variational Inference for Non-Autoregressive Text-to-Speech. [Paper](https://openreview.net/pdf?id=o3iritJHLfO) & [Code](https://github.com/LEEYOONHYUNG/BVAE-TTS)
- [NAACL] Align-Refine: Non-Autoregressive Speech Recognition via Iterative Realignment. [Paper](https://aclanthology.org/2021.naacl-main.154.pdf)
- [arxiv] VARA-TTS: Non-Autoregressive Text-to-Speech Synthesis based on Very Deep VAE with Residual Attention. [Paper](https://arxiv.org/pdf/2102.06431v1.pdf)
- [arvix] Fast End-to-End Speech Recognition via a Non-Autoregressive Model and Cross-Modal Knowledge Transferring from BERT. [Paper](https://arxiv.org/pdf/2102.07594)
- [ASRU][ Non-autoregressive Mandarin-English Code-switching Speech Recognition with Pinyin Mask-CTC and Word Embedding Regularization. [Paper](https://arxiv.org/pdf/2104.02258)
- [arxiv] Pushing the Limits of Non-Autoregressive Speech Recognition. [Paper](https://arxiv.org/pdf/2104.03416v4.pdf)
- [arxiv] WNARS: WFST based Non-autoregressive Streaming End-to-End Speech Recognition. [Paper](https://arxiv.org/pdf/2104.03587v2.pdf)
- [arxiv] TalkNet 2: Non-Autoregressive Depth-Wise Separable Convolutional Model for Speech Synthesis with Explicit Pitch and Duration Prediction. [Paper](https://arxiv.org/pdf/2104.08189v3.pdf) & [Code](https://github.com/rishikksh20/TalkNet2-pytorch)
- [Interspeech] An Improved Single Step Non-autoregressive Transformer for Automatic Speech Recognition. [Paper](https://arxiv.org/pdf/2106.09885v2.pdf)
- [Speech Synthesis Workshop] Non-Autoregressive TTS with Explicit Duration Modelling for Low-Resource Highly Expressive Speech. [Ppaer](https://arxiv.org/pdf/2106.12896v2.pdf)
- [arxiv] VAENAR-TTS: Variational Auto-Encoder based Non-AutoRegressive Text-to-Speech Synthesis. [Paper](https://arxiv.org/pdf/2107.03298v1.pdf) & [Code](https://github.com/thuhcsi/VAENAR-TTS)
- [arxiv] Non-autoregressive End-to-end Speech Translation with Parallel Autoregressive Rescoring. [Paper](https://arxiv.org/pdf/2109.04411v1.pdf)
- [arxiv] Non-autoregressive Transformer with Unified Bidirectional Decoder for Automatic Speech Recognition. [Paper](https://arxiv.org/pdf/2109.06684v1.pdf)
- [ASRU] Fast-MD: Fast Multi-Decoder End-to-End Speech Translation with Non-Autoregressive Hidden Intermediates. [Paper](https://arxiv.org/pdf/2109.12804v1.pdf)
- [ASRU] A Comparative Study on Non-Autoregressive Modelings for Speech-to-Text Generation. [Paper](https://arxiv.org/pdf/2110.05249v1.pdf)
- [arxiv] Exploring Timbre Disentanglement in Non-Autoregressive Cross-Lingual Text-to-Speech. [Paper](https://arxiv.org/pdf/2110.07192v1.pdf) 
- [arxiv] Boundary and Context Aware Training for CIF-based Non-Autoregressive End-to-end ASR. [Paper](https://arxiv.org/pdf/2104.04702)
- [arxiv] Non-autoregressive Transformer-based End-to-end ASR using BERT. [Paper](https://arxiv.org/pdf/2104.04805v1.pdf)
- [Interspeech ] Multi-Speaker ASR Combining Non-Autoregressive Conformer CTC and Conditional Speaker Chain. [Paper](https://arxiv.org/pdf/2106.08595v1.pdf) & [Code](https://github.com/pengchengguo/espnet)
- [Interspeech] Streaming End-to-End ASR based on Blockwise Non-Autoregressive Models. [Paper](https://arxiv.org/pdf/2107.09428v1.pdf) & [Code](https://github.com/espnet/espnet)
### 2020
- [arXiv] Listen and Fill in the Missing Letters: Non-Autoregressive Transformer for Speech Recognition. [Paper](https://arxiv.org/pdf/1911.04908.pdf)
- [INTERSPEECH] Mask CTC: Non-Autoregressive End-to-End ASR with CTC and Mask Predict. [Paper](https://arxiv.org/pdf/2005.08700.pdf) & [Code](https://github.com/espnet/espnet)
- [INTERSPEECH] Insertion-Based Modeling for End-to-End Automatic Speech Recognition. [Paper](https://arxiv.org/pdf/2005.13211.pdf) & [Code](https://github.com/espnet/espnet)
- [ICML] Non-Autoregressive Neural Text-to-Speech [Paper](https://arxiv.org/pdf/1905.08459.pdf) & [Code](https://github.com/ksw0306/WaveVAE)
- [INTERSPEECH] Listen Attentively, and Spell Once: Whole Sentence Generation via a Non-Autoregressive Architecture for Low-Latency Speech Recognition. [Paper](https://arxiv.org/pdf/2005.04862v4.pdf)
- [arxiv] Spike-Triggered Non-Autoregressive Transformer for End-to-End Speech Recognition. [Paper](https://arxiv.org/pdf/2005.07903v1.pdf)
- [Interspeech] Quasi-Periodic Parallel WaveGAN Vocoder: A Non-autoregressive Pitch-dependent Dilated Convolution Model for Parametric Speech Generation. [Paper](https://arxiv.org/pdf/2005.08654v1.pdf) & [Code](https://github.com/bigpon/QPPWG)
- [arxiv] Non-Autoregressive Predictive Coding for Learning Speech Representations from Local Dependencies. [Paper](https://arxiv.org/pdf/2011.00406v1.pdf) & [Code](https://github.com/Alexander-H-Liu/NPC)
- [arxiv] Hierarchical Prosody Modeling for Non-Autoregressive Speech Synthesis. [Paper](https://arxiv.org/pdf/2011.06465v3.pdf) & [Code](https://github.com/ming024/FastSpeech2)
### 2019
- [NeurIPS] FastSpeech: Fast, Robust and Controllable Text to Speech. [Paper](https://arxiv.org/pdf/1905.09263.pdf) & [Code](https://github.com/coqui-ai/TTS)

### Other tasks (Text Summarization; Dialogue and Intent Detection; Grammatical Error Correction; Text Style Transfer; Parsing; etc.)
- [21ACL-IJCNLP Findings] A Non-Autoregressive Edit-Based Approach to Controllable Text Simplification. [Paper](https://aclanthology.org/2021.findings-acl.330.pdf)
- [21ACL-IJCNLP Findings] NAST: A Non-Autoregressive Generator with Word Alignment for Unsupervised Text Style Transfer. [Paper](https://arxiv.org/pdf/2106.02210v1.pdf) & [Code](https://github.com/thu-coai/NAST)
- [21ACL-IJCNLP] POS-Constrained Parallel Decoding for Non-autoregressive Generation. [Paper](https://aclanthology.org/2021.acl-long.467.pdf) & [Code](https://github.com/yangkexin/pospd)
- [21ArXiv] Integrated Training for Sequence-to-Sequence Models Using Non-Autoregressive Transformer. [Paper](https://arxiv.org/pdf/2109.12950v1.pdf)
- [21ArXiv] Improving Non-autoregressive Generation with Mixup Training. [Paper](https://arxiv.org/pdf/2110.11115v1.pdf) & [Code](https://github.com/kongds/mist)
- [21EACL] Non-Autoregressive Text Generation with Pre-trained Language Models. [Paper](https://arxiv.org/pdf/2102.08220v1.pdf) & [Code](https://github.com/yxuansu/NAG-BERT)
- [21ArXiv] EncT5: Fine-tuning T5 Encoder for Non-autoregressive Tasks. [Paper](https://arxiv.org/pdf/2110.08426v1.pdf)
- [21ICML] BANG: Bridging Autoregressive and Non-autoregressive Generation with Large Scale Pretraining. [Paper](https://arxiv.org/pdf/2012.15525v3.pdf) & [Code](https://github.com/microsoft/BANG)
- [UnderReview] Learning Non-Autoregressive Models from Search for Unsupervised Sentence Summarization. [Paper](https://openreview.net/pdf?id=UNzc8gReN7m).
- [21ACL-IJCNLP] Tail-to-Tail Non-Autoregressive Sequence Prediction for Chinese Grammatical Error Correction. [Paper](https://arxiv.org/pdf/2106.01609v3.pdf) & [Code](https://github.com/lipiji/TtT)
- [WNUT] Character Transformations for Non-Autoregressive GEC Tagging. [Paper](https://arxiv.org/pdf/2111.09280v1.pdf) & [Code](https://github.com/ufal/wnut2021_character_transformations_gec)
- [21EMNLP] Exploring Non-Autoregressive Text Style Transfer. [Paper](https://aclanthology.org/2021.emnlp-main.730.pdf) & [Code](https://github.com/sunlight-ym/nar_style_transfer)
- [21EMNLP] Thinking Clearly, Talking Fast: Concept-Guided Non-Autoregressive Generation for Open-Domain Dialogue Systems. [Paper](https://arxiv.org/pdf/2109.04084v1.pdf) & [Code](https://github.com/rowitzou/cg-nar)
- [20ICLR] Non-Autoregressive Dialog State Tracking. [Paper](https://arxiv.org/pdf/2002.08024v1.pdf) & [Code](https://github.com/henryhungle/NADST)
- [20ArXiv] Non-Autoregressive Neural Dialogue Generation. [Paper](https://arxiv.org/pdf/2002.04250v2.pdf)
- [21ACL-IJCNLP] GL-GIN: Fast and Accurate Non-Autoregressive Model for Joint Multiple Intent Detection and Slot Filling. [Paper](https://arxiv.org/pdf/2106.01925v1.pdf) & [Code](https://github.com/yizhen20133868/GL-GIN)
- [20EMNLP] SlotRefine: A Fast Non-Autoregressive Model for Joint Intent Detection and Slot Filling. [Paper](https://arxiv.org/pdf/2010.02693v2.pdf) & [Code](https://github.com/moore3930/SlotRefine)
- [21EMNLP] Maximal Clique Based Non-Autoregressive Open Information Extraction. [Paper](https://aclanthology.org/2021.emnlp-main.764.pdf)
- [21EMNLP Findings] Span Pointer Networks for Non-Autoregressive Task-Oriented Semantic Parsing. [Paper](https://arxiv.org/pdf/2104.07275v3.pdf)
- [21NAACL-HLT] Non-Autoregressive Semantic Parsing for Compositional Task-Oriented Dialog. [Paper](https://arxiv.org/pdf/2104.04923v1.pdf) & [Code](https://github.com/facebookresearch/pytext)
- [20ArXiv] Recursive Non-Autoregressive Graph-to-Graph Transformer for Dependency Parsing with Iterative Refinement. [Paper](https://arxiv.org/pdf/2003.13118v2.pdf) & [Code](https://github.com/idiap/g2g-transformer)
- [21ArXiv] An Effective Non-Autoregressive Model for Spoken Language Understanding. [Paper](https://arxiv.org/pdf/2108.07005v1.pdf)
- [20ArXiv] A Study on the Autoregressive and non-Autoregressive Multi-label Learning. [Paper](https://arxiv.org/pdf/2012.01711v1.pdf)

