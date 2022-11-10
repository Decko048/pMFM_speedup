# pMFM_speedup

Use deep learning (DL) to speed up pMFM parameter selection/training

[TOC]

## Introduction

### pMFM

![pMFM_speedup-pMFM](https://user-images.githubusercontent.com/61874388/168448462-7efcb55a-93f0-4b68-8ad8-00484b76b4fb.png)

Proposed by the NUS Computational Brain Imaging Group (CBIG), the **parametric mean-ﬁeld model (pMFM)** is a computational model (not based on machine learning) that aims to simulate interactions between brain regions when individuals perform specific tasks.

In the context of this project, we consider 68 brain regions (i.e. cortical regions of interest (ROI)). Each human subject has a 68x68 brain structural connectivity (SC) matrix, indicating how strong the physical connections between different brain regions are. Each subject had also gone through the rs-fMRI procedure to generate the corresponding BOLD (Blood Oxygenation Level-Dependent) signal, which can reflect how active each brain regions are. The BOLD signal is then used to generate a 68x68 functional connectivity (FC) matrix (How strong are the correlations between activities of brain region pairs) and an 1118 × 1118 functional connectivity dynamics (FCD) matrix (how FC changes over time). Subjects are formed into groups, with corresponding group-averaged SC, FC, and FCD

The input to the pMFM includes a group-level SC, and 205 parameters (1 + 3 x 68 = 205): There is a global scaling factor G, and each brain region has three parameters wEE, wEI, and 𝜎. After taking in an SC and a parameter vector of length 205, the pMFM will generate a simulated BOLD signal, which is then used to generate simulated FC and FCD. Then, the simulated FC and FCD are compared with the empirical FC and FCD, which comes from the actual BOLD signals of subjects in the subject group. The metrics we use to compare the difference of these matrices are as follows:

- FC_CORR: FC correlation cost, defined by 1 - correlation between the simulated FC and the empirical FC
- FC_L1: The L1 distance between the simulated FC and the empirical FC
- FCD_KS: The KS statistic between the simulated FCD and the empirical FCD

The goal of the pMFM study is to get insights into how brain regions communicate with each other via analyzing the pMFM with parameters that can generate BOLD signals similar to the empirical ones. To find these "good" parameters, we need to take a subject group's SC and optimize the 205 parameters so that the pMFM can generate a BOLD signal with FC and FCD that are similar to the empirical ones.

### Motivation for pMFM_speedup

![pMFM_speedup-Deep Learning Model](https://s2.loli.net/2022/11/10/Rfrw8odSjcPNkVl.png)

The current pMFM has a major downside: the simulation is very slow. The underlying equations of pMFM are nonlinear ordinary differential equations (ODEs) that do not have a closed-form solution. Hence, the forward Euler method (a type of numerical procedure for solving ODEs) is used in pMFM, which has an undesirable simulation speed (15 mins for one run). Therefore, it requires a significant amount of time to find a good parameter vector. The goal of pMFM_speedup is to come up with a deep learning model to help the parameter selection process. The inputs to pMFM_speedup models are the same as that of pMFM and the pMFM_speedup models will perform a regression task to output a cost vector containing FC_CORR, FC_L1, and FCD_KS indicating how good the input parameter vector is. In this way, the pMFM_speedup can filter out bad parameters with much less time (compared to pMFM), and we only need to use pMFM for those parameters with good cost vectors.



## Dataset Creation

### Data Availability

The raw diffusion MRI, rs-fMRI, and T1w/T2w data are retrieved from the Human Connectome Project ([HCP](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release))

### Dataset Generation Process

<img width="500" alt="pMFM_speedup-Subject Groups" src="https://user-images.githubusercontent.com/61874388/171318768-818c5e3e-afff-4cf4-a003-267dcdcedf05.png">

- 1000 subjects are split into 88 groups
  - There are 50 subjects within each group
  - train set has 57 groups, validation set has 14 groups, and test set has 17 groups
    - Subjects are not shared between train/validation/test sets (to avoid **data snooping/leakage**)
    - Adjacent groups (e.g. train 1 and train 2) share 40 subjects
- Generate group-level SCs from individual SCs and get empirical FCs, empirical FCDs from empirical BOLD signals
- Feed the SC and selected parameters (see below) for each group into pMFM and generate pMFM FC and FCD
- Generate correlation FC_CORR and L1 distance FC_L1 between pMFM FC and the empirical one
- Generate KS statistics FCD_KS between pMFM FCD and the empirical one

#### Parameter Selection for each Subject Group

- Initialize CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with G, wEE, wEI, and 𝜎 in their respective ranges
  - Range for each wEE: [1, 10]
  - Range for each wEI: [1, 5]
  - Range for each 𝜎: [0.0005, 0.01]
  - Range for G: [0, 3]
- Use CMA-ES (100 iterations with each iteration yielding 100 children) to generate 10000 different parameter vectors for each subject group
- We will have 88 SCs, each with 10000 parameter vectors, resulting in 880,000 inputs

### Generated Dataset

To summarize, the generated dataset contains 880,000 inputs, each contains:

- Input features:

  - Group-level SC

  - wEE, wEI, and 𝜎 for each ROI

  - Global scaling factor G

- Target value/Ground truths:
  - Corresponding FC correlation, FC L1 distance, and FCD KS costs



## Assumption Validation

Below are some interesting findings about FC, FCD, SC and param vector.

### Assumption 1: Simulated FC and FCD are not sensitive to changes in SC

**Motivation**: The pair-wise correlations (Pearson correlation coefficient) between all group SCs are computed and they are highly similar. Hence changes in SC may not result in large variation in FC or FCD

<img width="500" alt="corr_between_all_SCs" src="https://user-images.githubusercontent.com/61874388/171318852-cc74542c-f641-4e34-9f74-dcaaab519dc0.png">

**Validation**: The validation process is described below:

- Randomly choose a parameter vector from the train/validation set
- Fix the parameter vector and use different SCs with that parameter vector to generate corresponding simulated FC and FCD
- For each pair of SC, compute correlation between SCs and the pair-wise FC correlation/FC L1 cost/FCD KS statistic

<img width="250" alt="costs_vs_SC_correlation" src="https://s2.loli.net/2022/11/10/uvUD7NKxJo9rgz6.png">

The results for one of the param vector are shown below:![image-20221110162803524](https://s2.loli.net/2022/11/10/EyDq69nSfLCKVP8.png)

![image-20221110162812616](https://s2.loli.net/2022/11/10/J5Pt34ExBDg78Fb.png)

![image-20221110162820430](https://s2.loli.net/2022/11/10/R2gSoKEW7XH6Pa1.png)

This validation process is repeated for different param vectors and the results are similar. As shown above, when SC changes, the simulated FC/FCD are quite similar (high FC correlation, low FC L1 cost, low FCD KS statistic). Hence, the simulated FC and FCD are not sensitive to changes in SC.

### Assumption 2: Simulated FC&FCD are not sensitive to changes in param vector

**Motivation**: After verifying the simulated FC and FCD are not sensitive to changes in SC, we are also interested in whether the similated FC and FCD are sensitive to changes in param vector

**Validation**: The validation process is very similar to that of SC:

- Randomly choose a SC from the train/validation set
- Fix the SC and use different parameter vectors with that SC to generate corresponding simulated FC and FCD
- For each pair of parameter vectors, compute correlation between parameter vectors and the pair-wise FC correlation/FC L1 cost/FCD KS statistic



<img width="250" alt="costs_vs_param_correlation" src="https://s2.loli.net/2022/11/10/aMtLw6XAzerRQZS.png">

The results for one of the param vector are shown below:

![image-20221110170219758](https://s2.loli.net/2022/11/10/hpGXD21OUwJknMx.png)

![image-20221110170230210](https://s2.loli.net/2022/11/10/sQCw5dkZjDSOmhX.png)

![image-20221110170235849](https://s2.loli.net/2022/11/10/zywpaSmMdeQBDNR.png)

This validation process is repeated for different param vectors and the results are similar. The above figures demonstrated that when param vector changes, the simulated FC/FCD are also quite similar (high FC correlation, low FC L1 cost, low FCD KS statistic). Hence, the simulated FC and FCD are not sensitive to changes in param vector.



## Models

### Common elements for model training:

- **Loss function**: Mean Square Error (MSE) Loss

- **Optimizer**: Adam optimizer (initial lr = 5e-4) with exponential decay learning rate scheduler (multiply lr by 0.98 every epoch)

- **Batch Size**: 256

- **Metrics logged**:
  - During **training**: mse loss across 3 different cost terms
  - During **validation**: mse loss across 3 different cost terms, and mse loss for each individual cost

- **Hyperparameter** **tuning**: Optuna (with max epoch equal to 100)

### SC Feature Extractor

Since the SC matrix has many parameters (68x68) and SC matrices from different groups are highly similar. We can use neural nets to extract features/embedding from the SC matrix and proceed further with the SC feature with a much lower dimension. In this way, we have much fewer input parameters, which can address overfitting to some extent.

Two SC feature extractors are provided:

- CNN (Convolutional Neural Network) version: we treat an SC matrix as a 68x68 image and use convolution to extract latent features of the SC matrix

  ![pMFM_speedup-Extract SC Feature CNN](https://user-images.githubusercontent.com/61874388/171318608-9f0e0af0-0045-446d-b3c9-56123c536341.png)

- MLP (Multi-Layer Perceptron) version: we first vectorize the upper triangular part of the SC (without the diagonal entries), and feed the SC vector into a MLP to extract features

  ![pMFM_speedup-Extract SC Feature MLP](https://s2.loli.net/2022/11/10/taW5dM4Vnbsgp7e.png)

The experiements below used MLP as the SC feature extractor

### Experimented Models

#### Naive Net

![pMFM_speedup-Naive Net](https://user-images.githubusercontent.com/61874388/171318654-6f3837a8-d603-436c-a39f-306427afd5bc.png)

Naive net is a straightforward simple MLP network with several ReLU activated linear layers. If the SC feature is extracted and used as a part of the input, it will be concatenated with param vector.

### Naive Net with Coefficient Vector

![pMFM_speedup-Naive_Net_with_Coef](https://s2.loli.net/2022/11/10/w4BxmpId83AWFUD.png)

Note that the wEE, wEI, and 𝜎 for each ROI can be derived using 9 coefficients, group-level myelin and group-level RSFC gradient (wEE, wEI, and 𝜎 have been parameterized). Hence, we have a variant of the naive net, where we use the Coefficient Vector (9 linear coefficients and the global scaling factor G) as the input.

#### Plain GCN

![pMFM_speedup-GCN](https://user-images.githubusercontent.com/61874388/171318686-eb8f1e37-1fc9-4e87-a200-8d2bb5b02698.png)

We can treat each brain region as a node, and the parameters associated with a brain region as the corresponding node features. The edge and edge weight come from non-zero entries of a group's SC. The global scaling factor G is used to scale the edge weight, similar to its role in the pMFM (scaling connections between brain regions uniformly).



## Experiment Results

Each of the model is tunned by optuna: 100 set of hyperparmeters have been tried for each model, each trial's max number of epoch is 100. Then the top 10 set of hyperparameters are picked based on validation cost, and their performances (total MSE loss, which is mathematically equivalent to the mean of 3 cost terms' MSEs) are shown in the box plot below.

<img width="400" alt="mse_loss_box_plot" src="https://s2.loli.net/2022/11/10/9C3JVch6tjXxgMl.png">

We can see that Naive Net without SC performs the best (has the lowest validation loss and the lowest variation)

The MSE losses for each cost term are shown below:

|                        FC_CORR’s MSE                         |                         FC_L1’s MSE                          |                         FCD_KS’s MSE                         |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![FC_CORR’s MSE](https://s2.loli.net/2022/11/10/PykjoqO3VZKX8MC.png) | ![FC_L1’s MSE](https://s2.loli.net/2022/11/10/ifYvMbZ6d7k2mX1.png) | ![FCD_KS’s MSE](https://s2.loli.net/2022/11/10/kAT8wPhGNSqBuV7.png) |

We can see that the individual MSE is consistent with the total MSE. And Naive Net without SC performs consistently better than other models.

The best model (Naive Net without SC) is then tested on the test set, and the MSE loss is 0.0248.

### Compare prediction and actual costs

We then evaluate the best model's (Naive Net without SC) performance on each test subject group by comparing the predictions (output from DL model) and the actual costs (the ground truth costs derived from comparing pMFM FC and FCD with the empirical FC and FCD). The results for total costs (sum of the 3 cost terms) are shown below. The figure on the left shows predicted total costs and the actual total costs for one test group (dots with an actual cost of 3 represent bad parameters with excitatory firing rate outside 2.7 Hz and 3.3Hz when fed into pMFM). The distribution on the right shows the correlations between the predicted total costs and the actual total costs for different test groups.

| Predicted total costs and actual total costs for one test group | Distribution of correlation between predicted total costs and actual total costs in the test set |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![Predicted costs and actual costs for one test group ](https://s2.loli.net/2022/11/10/B5LZ48UDNMuAJGd.png) | ![image-20221110191738050](https://s2.loli.net/2022/11/10/fzZVhcU5S8dl9FC.png) |

The comparison between the prediction and actual cost is also performed for individual cost terms and the results can be found in `reports/testing/compare_top_k_params/basic_models/naive_net/no_SC`. 

Overall, the predictions and actual costs are strongly correlated, which shows the effectiveness of the DL model.

## **Parameter optimization** using DL model

