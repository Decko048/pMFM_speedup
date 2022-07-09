# pMFM_speedup

Use deep learning to speed up pMFM parameter selection/training

## Introduction

#### pMFM

![pMFM_speedup-pMFM](https://user-images.githubusercontent.com/61874388/168448462-7efcb55a-93f0-4b68-8ad8-00484b76b4fb.png)

Proposed by the NUS Computational Brain Imaging Group (CBIG), the **parametric mean-ﬁeld model (pMFM)** is a computational model (not based on machine learning) that aims to simulate interactions between brain regions when individuals perform specific tasks.

In the context of this project, we consider 68 brain regions (i.e. cortical regions of interest (ROI)). Each human subject has a 68x68 brain structural connectivity (SC) matrix, indicating how strong the physical connections between different brain regions are. Each subject had also gone through the rs-fMRI procedure to generate the corresponding BOLD (Blood Oxygenation Level-Dependent) signal, which can reflect how active each brain regions are. The BOLD signal is then used to generate a 68x68 functional connectivity (FC) matrix (How strong are the correlations between activities of brain region pairs) and an 1118 × 1118 functional connectivity dynamics (FCD) matrix (how FC changes over time). Subjects are formed into groups, with corresponding group-averaged SC, FC, and FCD

The input to the pMFM includes a group-level SC, and 205 parameters (1 + 3 x 68 = 205): There is a global scaling factor G, and each brain region has three parameters wEE, wEI, and 𝜎. After taking in an SC and a parameter vector of length 205, the pMFM will generate a simulated BOLD signal, which is then used to generate simulated FC and FCD. Then, the simulated FC and FCD are compared with the empirical FC and FCD, which comes from the actual BOLD signals of subjects in the subject group (the metrics we use to compare the difference of these matrices are the correlation of FCs, the L1 distance between FCs, and the KS statistic of FCDs). The goal of the pMFM study is to get insights into how brain regions communicate with each other via analyzing the pMFM with parameters that can generate BOLD signals similar to the empirical ones. To find these "good" parameters, we need to take a subject group's SC and optimize the 205 parameters so that the pMFM can generate a BOLD signal with FC and FCD that are similar to the empirical ones.

#### Motivation for pMFM_speedup

![pMFM_speedup-Deep Learning Model](https://user-images.githubusercontent.com/61874388/168448463-137c9670-9dd9-4d2a-8f59-48ca3b70888e.png)

The current pMFM has a major downside: the simulation is very slow. The underlying equations of pMFM are nonlinear ordinary differential equations (ODEs) that do not have a closed-form solution. Hence, the forward Euler method (a type of procedure for solving ODEs) is used in pMFM, which has an undesirable simulation speed (15 mins for one run). Therefore, it requires a significant amount of time to find a good parameter vector, since each trial takes 15 mins. The goal of pMFM_speedup is to come up with a deep learning model to help the parameter selection process. The inputs to pMFM_speedup models are the same as that of pMFM and the pMFM_speedup models will perform a classification task and output either GOOD, OK, or BAD (or simply GOOD or BAD for binary classification), indicating how good the input parameter vector is. In this way, the pMFM_speedup can filter out bad parameters with much less time (compared to pMFM), and we only need to use pMFM for those parameters classified as GOOD.

## Dataset Creation

#### Dataset Generation Process

<img width="500" alt="pMFM_speedup-Subject Groups" src="https://user-images.githubusercontent.com/61874388/171318768-818c5e3e-afff-4cf4-a003-267dcdcedf05.png">

- 1000 subjects are split into 88 groups (cannot randomly select groups for 88 times because it may result in **data snooping/leakage** since we are using group-level SC)

  - There are 50 subjects within each group
  - train split has 57 groups, validation split has 14 groups, and test split has 17 groups.
    - Splits do not share subjects (to avoid **data snooping/leakage**)
    - Adjacent groups (e.g. train 1 and train 2) share 40 subjects
- Generate group-level SCs from individual SCs and get empirical FCs, empirical FCDs from empirical BOLD signals
- Feed SCs and selected parameters (see below) into pMFM and generate pMFM FC and FCD
- Generate correlation FC_CORR and L1 distance FC_L1 between pMFM FC and the empirical one
- Generate KS statistics FCD_KS between pMFM FCD and the empirical one

##### Parameter Selection for Data Generation

- Use CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to select G, wEE, wEI, and 𝜎 in their respective ranges, and generate 10000 different parameter vectors for each subject group
  - Ranges
    - Range for each wEE: [1, 10]
    - Range for each wEI: [1, 5]
    - Range for each 𝜎: [0.0005, 0.01]
    - Range for G: [0, 3]
- We will have 88 SCs and each with 10000 parameter vectors, resulting in 880,000 inputs

#### Generated Dataset

The generated dataset contains 880,000 inputs, each contains:

- Group-level SC
- wEE, wEI, and 𝜎 for each ROI
- Global scaling factor G
- Corresponding FC correlation, FC L1 distance, and FCD KS costs.

## Input, ground truth, and output of deep learning models:

💡 **Note**: we are predicting the quality of parameters for different cost types (FC_CORR, FC_L1, FCD_KS) individually (i.e., we may use three different models, one for each cost type). In this way, we don't need to have a pre-determined way of how to combine FC_CORR, FC_L1 and FCD_KS to get the cost function, which may not have the same importance/weight. After three models are trained, we can rely on the three predictions and judge whether the parameters are worthy enough to be put into the slow pMFM simulation (currently we deem a parameter worthy enough if all three predictions are GOOD).

**Input:**

- 68x68 SC
- For every ROI we have wEE, wEI, and 𝜎: in total, 3x68 = 204 parameters
- Global parameter G

**Ground truth label:**

Based on pre-determined thresholds, each input is associated with a label (GOOD, OK, BAD) for a given cost type (FC_CORR, FC_L1, FCD_KS).

*Thresholds (tri-class):*

| Cost Type | GOOD threshold |     OK threshold     | BAD threshold |
| :-------: | :-------------: | :-------------------: | :------------: |
|  FC_CORR  | FC_CORR ≤ 0.35 | 0.35< FC_CORR ≤ 0.45 | FC_CORR > 0.45 |
|   FC_L1   |  FC_L1 ≤ 0.08  |  0.08 < FC_L1 ≤ 0.2  |  FC_L1 > 0.2  |
|  FCD_KS  |  FCD_KS ≤ 0.2  |  0.2 < FCD_KS ≤ 0.4  |  FCD_KS > 0.4  |

*Thresholds (bi-class):* essentially treating OK parameters as BAD as well

| Cost Type | GOOD threshold | BAD threshold |
| :-------: | :-------------: | :------------: |
|  FC_CORR  | FC_CORR ≤ 0.35 | FC_CORR > 0.35 |
|   FC_L1   |  FC_L1 ≤ 0.08  |  FC_L1 > 0.08  |
|  FCD_KS  |  FCD_KS ≤ 0.2  |  FCD_KS > 0.2  |

**Output:**

For a given cost type (FC_CORR, FC_L1, FCD_KS), classify the input as either GOOD, OK, or BAD (or simply GOOD or BAD for binary classification).

## Assumption Validation

### Assumption 1: The thresholds are reasonable

**Motivation**: we have thresholds to decide whether a parameter is GOOD for a given cost type. Hence, it is of great importance to examine the thresholds, especially the costs distributions of all GOOD parameters (all three costs are labeled GOOD) using these thresholds.

**Validation**: the costs distributions of all GOOD parameters from different split are plotted for each split (here, the total cost = FC_CORR + FC_L1 + FCD_KS):
![all_GOOD_params_costs-train](https://user-images.githubusercontent.com/61874388/171319909-0a952d68-d940-488f-bf67-e9882f5827e1.png)

![all_GOOD_params_costs-validation](https://user-images.githubusercontent.com/61874388/171319912-cdba3f3b-2553-4182-84d2-3c4ee79d382f.png)

![all_GOOD_params_costs-test](https://user-images.githubusercontent.com/61874388/171319916-6ebbd4c4-2c68-4427-852d-578b2075e0f8.png)

As we can see, distributions of FC_L1 cost and FCD_KS cost are skewed towards the threshold, and we were able to capture most GOOD parameters with the current threshold (if we push the threshold further to the left, many relatively GOOD parameters will be lost). As for FC_CORR cost, it may seem like we can further decrease the FC_CORR GOOD threshold. However, as can be seen from the above FC_CORR costs distributions of all GOOD parameters, the distributions and means of distributions are different for different splits. Consequently, the FC_CORR label distribution may become unstable across splits if the GOOD threshold is decreased and the model trained with the decreased threshold might not generalize well to unseen parameters.

*Example: FC_CORR label distribution across splits when the GOOD FC_CORR threshold is 0.29*
`<img width="1000" alt="FC_CORR_label_distribution_GOOD_threshold_0 29" src="https://user-images.githubusercontent.com/61874388/171318484-21e77d29-6ae5-4b72-a316-55feaaa8b0cb.png">`

In contrast, the FC_CORR label distribution is more stable across splits when the GOOD threshold is 0.35 (Notice that OK class becomes highly imbalanced, which will impact the group-averaged metrics for tri-class classification).

*FC_CORR label distribution across splits when the GOOD FC_CORR threshold is 0.35*
`<img width="1000" alt="FC_CORR_label_distribution_GOOD_threshold_0 35" src="https://user-images.githubusercontent.com/61874388/171318493-6ae8dee4-3e7e-4002-87b5-dc8c5a7e4acd.png">`

Note that the FC_L1 and FCD_KS label distributions are more stable across splits as well using the current threshold.

Hence, the current thresholds reasonably fit the task.

### Assumption 2: SCs do not contribute much to the predictions

**Motivation**: The pair-wise correlations (Pearson correlation coefficient) between all group SCs are computed and they are highly similar. Hence it may not contribute much to the prediction:

<img width="500" alt="corr_between_all_SCs" src="https://user-images.githubusercontent.com/61874388/171318852-cc74542c-f641-4e34-9f74-dcaaab519dc0.png">

**Validation**:

Firstly, we try to take a group's SC and use the parameters from another group and see if those parameters perform similarly (here, the total cost = FC_CORR + FC_L1 + FCD_KS).

*Example - use parameters from group test 5 and use them with group train 1's SC, and compare with the original costs of those parameters for test 5:*
![use_test_5_param_on_train_1_SC](https://user-images.githubusercontent.com/61874388/171319648-91df8dc9-aaf9-44b7-8cee-a7b6f7b095ab.png)

*Example - use parameters from group validation 1 and use them with group train 1's SC, and compare with the original costs of those parameters for validation 1:*
![use_validation_1_param_on_train_1_SC](https://user-images.githubusercontent.com/61874388/171319655-247db930-e735-4c59-9eb4-3ceca0d22bda.png)

The results show that the majority of the total cost difference comes from FC-related costs. FCD_KS's cost difference is generally small, sometimes even negative (using the same parameters with a different SC yield better FCDs). Additionally, the mean of total cost difference distribution generally lies in the range of 0.01 to 0.02 (with few exceptions like the two above), which shows that good parameters from a group can be generalized to another group with a different SC.

Moreover, the models are trained with and without SC features and the performances are not improving when SC features are added (see below).

Hence, SC features will not be used in the final models as it does not play a major role in prediction and they will add many more parameters to the models, which might lead to overfitting

### Assumption 3: parameters with good FCD_KS generally have good FC_CORR and FC_L1

**Motivation**: It is generally harder to find parameters with good FCD_KS. Therefore, it is possible that parameters with good FCD_KS will have good FC_CORR and FC_L1 as well

**Validation**:

This assumption is not true for many groups:

|  Group name  | #  GOOD FC_CORR | #  GOOD FC_L1 | #  GOOD FCD_KS | # params whose three costs are all GOOD |
| :-----------: | :-------------: | :-----------: | :------------: | :-------------------------------------: |
|   train  1   |      5052      |     2923     |      1737      |                   551                   |
|   train 11   |      5802      |     3159     |      1500      |                   895                   |
|   train 21   |      5516      |     1830     |      1638      |                    0                    |
|   train 31   |      6872      |     2827     |      1893      |                   253                   |
|   train 41   |      3043      |      974      |      1121      |                    0                    |
|   train 51   |      3956      |     2078     |      811      |                   106                   |
| validation 1 |      4993      |     2056     |      737      |                   17                   |
| validation 5 |      6862      |     3316     |      1064      |                   99                   |
| validation 10 |      4183      |     1787     |      891      |                    3                    |
|    test 1    |      7542      |     3792     |      3351      |                  1632                  |
|    test 5    |      5752      |     2752     |      811      |                   156                   |
|    test 10    |      5188      |     2591     |      1066      |                   85                   |
|    test 15    |      5223      |     2350     |      1056      |                   116                   |

The corresponding ratios of the "all GOOD" params in GOOD params for each cost type are:

|  Group name  | FC_CORR ratio | FC_L1 ratio | FCD_KS ratio |
| :-----------: | :-----------: | :---------: | :----------: |
|   train  1   |    0.1091    |   0.1885   |    0.3172    |
|   train 11   |    0.1543    |   0.2833   |    0.5967    |
|   train 21   |    0.0000    |   0.0000   |    0.0000    |
|   train 31   |    0.0368    |   0.0895   |    0.1337    |
|   train 41   |    0.0000    |   0.0000   |    0.0000    |
|   train 51   |    0.0268    |   0.0510   |    0.1307    |
| validation 1 |    0.0034    |   0.0083   |    0.0231    |
| validation 5 |    0.0144    |   0.0299   |    0.0930    |
| validation 10 |    0.0007    |   0.0017   |    0.0034    |
|    test 1    |    0.2164    |   0.4304   |    0.4870    |
|    test 5    |    0.0271    |   0.0567   |    0.1924    |
|    test 10    |    0.0164    |   0.0328   |    0.0797    |
|    test 15    |    0.0222    |   0.0494   |    0.1098    |

we can see that although ratio of "all GOOD" parameters among parameters with GOOD FCD_KS is higher than that of FC_CORR and FC_L1, the ratio flucutates across groups (some even have 0 "all GOOD" parameters). Therefore, we should not solely rely on FCD_KS predictor.

## Models

### Common elements for all models:

- **Loss function**: Cross-Entropy Loss
- **Optimizer**: Adam optimizer with early stopping (monitors validation f1 and stops training when the validation f1 stops increasing for 5 epochs)
- **Regularization**: batch normalization is used to reduce overfitting

### SC Feature Extractor

Since the SC matrix has many parameters (68x68) and SC matrices from different groups are highly similar. We can use neural nets to extract features ("embedding") from the SC matrix and proceed further with the SC feature with a much lower dimension. In this way, we have much fewer input parameters, which can address overfitting to some extent.

![pMFM_speedup-Extract SC Feature CNN](https://user-images.githubusercontent.com/61874388/171318608-9f0e0af0-0045-446d-b3c9-56123c536341.png)

The current SC feature extractor used is CNN, using which we treat an SC matrix as a 68x68 image and use convolution to extract latent features of the SC matrix.

### Sequence Feature Extractor

The sequence feature extractor is created with the intent to simulate how the brain state changes over time. The idea is that each sequence feature represents the brain state at some point in time. This aims to capture the FC dynamics (FCD, whose performance is known to be difficult to predict) over time.

There are two types of sequence feature extractors: one is formed by several linear layers, and the sequential features are obtained via extracting intermediate outputs of linear layers

<img width="700" alt="pMFM_speedup-FC Sequence Feature Extractor" src="https://user-images.githubusercontent.com/61874388/171323942-19dc35b2-c645-4e86-ab92-76cd0c54e234.png">

Another one uses GCN layers instead of  linear layers and obtains sequential features via extracting intermediate outputs of GCN layers.

<img width="700" alt="pMFM_speedup-GCN Sequence Feature Extractor" src="https://user-images.githubusercontent.com/61874388/171323950-01cf5c59-3324-4450-ae78-dae3fedea417.png">

### Experimented Models

Below is a summary of experimented models
![pMFM_speedup-Model](https://user-images.githubusercontent.com/61874388/171185184-efcd6e0d-c490-4b3d-9019-9d8dc386bea5.png)

#### Naive Net

![pMFM_speedup-Naive Net](https://user-images.githubusercontent.com/61874388/171318654-6f3837a8-d603-436c-a39f-306427afd5bc.png)

Naive net is a straightforward simple network with several ReLU activated linear layers.

#### Plain GCN

![pMFM_speedup-GCN](https://user-images.githubusercontent.com/61874388/171318686-eb8f1e37-1fc9-4e87-a200-8d2bb5b02698.png)

We can treat each brain region as a node, and the parameters associated with a brain region as the corresponding node features. The edge and edge weight come from non-zero entries of a group's SC. The global scaling factor G is used to scale the edge weight, similar to its role in the pMFM (scaling connections between brain regions uniformly).

#### Sequence Models

To simulate brain state changes across time, we extract some sequential features and input them into sequence models such as LSTM and Transformer encoder.

<img width="500" alt="pMFM_speedup-LSTM" src="https://user-images.githubusercontent.com/61874388/171323869-95f2534d-9168-420d-95dc-5d60f48b6d52.png">

For LSTM, the output of the last LSTM unit is fed into a linear layer for classification.

<img width="500" alt="pMFM_speedup-Transformer" src="https://user-images.githubusercontent.com/61874388/171323905-65551b3e-e03e-47e4-a153-075c5ccbe930.png">

For the Transformer encoder, outputs are concatenated and fed into a linear lay for classification.

## Experiments

### Common elements for all experiments:

- **Batch Size**: 256
- **Metrics logged**: Since we are mainly focusing on parameters with GOOD label, precision and recall of GOOD label are logged. Additionally, macro-averaged accuracy and f1 are also logged to provide more information on models' performances

  - During **training**: training accuracy, training f1, training loss
  - During **validation**: validation GOOD label precision & recall, validation accuracy, validation f1, validation loss
  - During **testing**: testing GOOD label precision & recall, testing accuracy, testing f1
- **Baseline accuracy**: the accuracy when we simply classify all instances as the majority class

  - | FC_CORR baseline  accuracy | FC_L1 baseline accuracy | FCD_KS baseline accuracy |
    | -------------------------- | ----------------------- | ------------------------ |
    | 51.32%                     | 54.19%                  | 71.96%                   |

### Tri-class Classification Performance

**Hyperparameter Settings**:

- SC feature extractor
  - The CNN used to extract SC features has three convolution layers (with output channels 8, 16, 32 respectively), each ReLU activated and Max Pooled (2x2 window). Finally, it has a linear layer to get the desired SC feature dimension (in this case 200)
- Sequence feature extractor
  - sequence models have sequence length of 5, and the dimension of sequence input is 100
  - GCN layer's hidden node feature is of dimension 5
- Models
  - Naive Net has 3 linear layers (first linear layer's output dimension is 160, second linear layer's output dimension is 64)
  - LSTM model uses 2 linear layers as classification layers after the output of the last LSTM unit, the output dimension of the first FC layer is 32
  - Transformer Encoder model uses 3 linear layers as classification layers (the first linear layer's output dimension is 160, the second linear layer's output dimension is 64). The Transformer Encoder used has 5 attention heads
  - Plain GCN has 5 GCN layers with hidden node feature of dimension 5 and 3 linear layers as classification layers (first linear layer's output dimension is 200, second linear layer's output dimension is 64)

#### FC_CORR

The performances of models when the cost type is FC_CORR:

| Models                           | Precision of GOOD label | Recall of GOOD label | Macro-average F1 | Macro-average Accuracy |
| -------------------------------- | ----------------------- | -------------------- | ---------------- | ---------------------- |
| Naïve Net with SC features      | 85.12%                  | 92.99%               | 79.13%           | 81.14%                 |
| Naïve Net without SC features   | 94.49%                  | 81.24%               | 82.68%           | 82.04%                 |
| LSTM with SC features            | 85.88%                  | 97.64%               | 83.48%           | 83.74%                 |
| LSTM without SC features         | 86.02%                  | 96.32%               | 83.29%           | 86.38%                 |
| Transformer with SC features     | 91.28%                  | 90.11%               | 84.64%           | 87.13%                 |
| Transformer without SC  features | 87.78%                  | 96.87%               | 86.06%           | 85.93%                 |
| Plain GCN                        | 65.94%                  | 60.15%               | 45.71%           | 46.61%                 |
| LSTM with GCN                    | 65.95%                  | 21.16%               | 38.41%           | 43.86%                 |
| Transformer with GCN             | 72.84%                  | 36.92%               | 40.40%           | 42.68%                 |

#### FC_L1

The performances of models when the cost type is FC_L1:

| Models                           | Precision of GOOD label | Recall of GOOD label | Macro-average F1 | Macro-average Accuracy |
| -------------------------------- | ----------------------- | -------------------- | ---------------- | ---------------------- |
| Naïve Net with SC features      | 73.30%                  | 85.83%               | 78.87%           | 80.13%                 |
| Naïve Net without SC features   | 63,22%                  | 83.60%               | 72.38%           | 74.48%                 |
| LSTM with SC features            | 59.55%                  | 88.10%               | 69.56%           | 72.29%                 |
| LSTM without SC features         | 82.39%                  | 77.23%               | 78.68%           | 78.03%                 |
| Transformer with SC features     | 82.13%                  | 75.73%               | 80.48%           | 80.51%                 |
| Transformer without SC  features | 80.86%                  | 82.31%               | 81.18%           | 81.19%                 |
| Plain GCN                        | 45.06%                  | 60.47%               | 53.69%           | 54.97%                 |
| LSTM with GCN                    | 18.04%                  | 3.59%                | 29.97%           | 35.08%                 |
| Transformer with GCN             | 1.60%                   | 0.23%                | 29.84%           | 34.97%                 |

#### FCD_KS

The performances of models when the cost type is FCD_KS:

| Models                           | Precision of GOOD label | Recall of GOOD label | Macro-average F1 | Macro-average Accuracy |
| -------------------------------- | ----------------------- | -------------------- | ---------------- | ---------------------- |
| Naïve Net with SC features      | 23.62%                  | 7.90%                | 40.68%           | 40.31%                 |
| Naïve Net without SC features   | 36.45%                  | 14.13%               | 54.09%           | 54.38%                 |
| LSTM with SC features            | 40.51%                  | 35.81%               | 59.99%           | 60.40%                 |
| LSTM without SC features         | 48.51%                  | 30.27%               | 60.38%           | 58.24%                 |
| Transformer with SC features     | 51.85%                  | 46.09%               | 66.20%           | 65.77%                 |
| Transformer without SC  features | 47.95%                  | 44.21%               | 52.58%           | 52.49%                 |
| Plain GCN                        | 22.75%                  | 1.75%                | 34.38%           | 36.36%                 |
| LSTM with GCN                    | 3.90%                   | 1.22%                | 26.85%           | 30.62%                 |
| Transformer with GCN             | 0.00%                   | 0.00%                | 31.19%           | 32.13%                 |

**Analysis**:

- Transformer and LSTM have similar performance (Transfomer has some slight edge)
- The simple Naive Net with three linear layers does not fall behind much
- SC features generally won't improve the performance (sometimes even hurt the performance)
- GCN models have bad performances, which may be due to underfitting (can consider adding more hidden features for each node)
- After examining the validation metrics, most models' performances have high variance. Although the bias for FC_CORR and FC_L1 is relatively good, the bias for FCD_KS still has room for improvement

### Bi-class Classification Performance

**Hyperparameter Settings**: same as the hyperparameters for the tri-class classification

#### FC_CORR

The performances of models when the cost type is FC_CORR:

| Models                           | Precision of GOOD label | Recall of GOOD label | F1     | Accuracy |
| -------------------------------- | ----------------------- | -------------------- | ------ | -------- |
| Naïve Net with SC features      | 92.48%                  | 92.81%               | 92.43% | 92.43%   |
| Naïve Net without SC features   | 89.26%                  | 94.45%               | 91.29% | 91.23%   |
| LSTM with SC features            | 97.87%                  | 75.98%               | 86.72% | 87.12%   |
| LSTM without SC features         | 84.47%                  | 98.17%               | 89.67% | 89.57%   |
| Transformer with SC features     | 94.39%                  | 92.38%               | 93.27% | 93.30%   |
| Transformer without SC  features | 94.30%                  | 90.97%               | 92.54% | 92.59%   |

#### FC_L1

The performances of models when the cost type is FC_L1:

| Models                           | Precision of GOOD label | Recall of GOOD label | F1     | Accuracy |
| -------------------------------- | ----------------------- | -------------------- | ------ | -------- |
| Naïve Net with SC features      | 80.06%                  | 61.06%               | 80.87% | 78.34%   |
| Naïve Net without SC features   | 53.36%                  | 83.66%               | 75.55% | 81.29%   |
| LSTM with SC features            | 85.08%                  | 76.33%               | 87.60% | 86.24%   |
| LSTM without SC features         | 54.05%                  | 98.13%               | 77.89% | 87.04%   |
| Transformer with SC features     | 79.13%                  | 70.45%               | 83.85% | 82.55%   |
| Transformer without SC  features | 74.94%                  | 74.38%               | 83.69% | 83.61%   |

#### FCD_KS

The performances of models when the cost type is FCD_KS:

| Models                           | Precision of GOOD label | Recall of GOOD label | F1     | Accuracy |
| -------------------------------- | ----------------------- | -------------------- | ------ | -------- |
| Naïve Net with SC features      | 45.15%                  | 13.70%               | 57.47% | 55.82%   |
| Naïve Net without SC features   | 27.26%                  | 4.33%                | 50.59% | 51.45%   |
| LSTM with SC features            | 44.33%                  | 39.19%               | 67.43% | 66.56%   |
| LSTM without SC features         | 53.93%                  | 41.07%               | 70.46% | 68.37%   |
| Transformer with SC features     | 51.41%                  | 42.31%               | 70.23% | 68.69%   |
| Transformer without SC  features | 57.40%                  | 41.48%               | 71.37% | 68.84%   |

**Analysis**:

- the precision and recall of GOOD label are similar to that of tri-class classification. However, f1 and accuracy are improved as expected since there are fewer classes for binary classification

## Going Foward

Use more hidden node features and retrain GCN models

Currently, the models' performances for all three costs have **high variance** => try to use bagging on strong learners (LSTM and Transformer) to reduce variance

Currently, the model's performance for FCD_KS have **high bias** => try to use gradient boosted weak leaners (naive net, decision stumps, or even simpler naive net) to reduce bias
