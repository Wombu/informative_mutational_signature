Explainable AI Model Reveals Informative Mutational 
Signatures for Cancer-Type Classification

Authors: Jonas Wagner, Jan Oldenburg, Neetika Nath and Stefan Simm

DOI: ...


Simple Summary: The objective of this research is to enhance the prediction of cancer types using an explainable artificial intelligence (XAI) model based on an artificial neural network with layerwise relevance propagation to extract informative mutational signatures. Multiple XAI models have been optimized using 10-fold cross-validation and grid search. In contrast to earlier approaches, the study compares the prediction capacities of unsupervised and supervised approaches. As outcomes, the paper showed better cancer-type-prediction accuracies using whole genome or intronic/intergenic mutation information instead of exome regions alone. Furthermore, the usage of mutational signatures is more relevant for prediction than localization information or driver gene mutation information. Overall, the XAI models developed in this study enabled informative mutational signatures to be generated for cancer-type and primary-site classification, leading to the detection of differences in the mechanistic characteristics of cancer types. These informative mutational signatures can be used in the future to more accurately and robustly diagnose cancer types as well as a foundation from which to identify new potential biomarkers and their context of impaired repair mechanisms.



Used Packages:
- python 3.11.7
- numpy 1.26.3
- torch 2.2.0
- pandas 2.1.4
- zennit 0.5.1
- sklearn 1.2.2
- matplotlib 3.8.4
- seaborn 0.13.2

  
Files

    main_WGS_GeneM.py

    main_WGS_MS+Bins.py

These Python scripts load their respective datasets (data/WGS_GeneM and data/WGS_MS+Bins), apply the validation splits, and compute both the LRP-epsilon values and the quantitative LRP results for all features. The outputs are saved to the designated output directory.

Note:
We provide two separate scripts due to differences in the exported ANN architectures' programming styles, even though their underlying logic and functionality are equivalent.
