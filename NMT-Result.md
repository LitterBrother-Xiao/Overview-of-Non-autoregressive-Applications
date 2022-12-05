### We show the performance on several datesets without rescoring reported from original paper. * indicates training with knowledge distillation from a big Transformer; ^ denotes training without sequence-level knowledge distillation; # refers to results on IWSLT'16 dataset. 
 
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
