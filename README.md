# gbm vs xgboost comparison, Credit Card Fraud Detection Binary Classification<br/>
Anonymized credit card transactions labeled as fraudulent or genuine.

## Experiment Environment<br/>
1. 27 core, 71 processors (Intel(R) Xeon(R) CPU @ 2.10GHz)
2. 528g total memory
3. Gbm (v‘2.1.3’) vs XGBboost(v‘0.81.0.1’) 

## Experiment Setup<br/>
[Download Data from Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
1. Total dataset = 284,807 x 31 with last column being binary outcome.
2. 70% training and 30% testing.
3. baseline gbm vs three variations of xgboost

## Example Results<br/>
### xgboost ROC<br/>
![](img/roc_xgboost.png)
### gbm ROC<br/>
![](img/roc_gbm.png)

### Time<br/>
gbm: 4 mins<br/>
xgboost: 74s-76s<br/>
Conclusion: xgboost is clear winner.<br/>

### AUC<br/>
gbm: 0.961<br/>
xgboost: 0.9644-0.967<br/>
Conclusion: xgboost is sightly better.<br/>

### Feature Importance<br/>
Feature gbm_fi<br/>
1: V12 52.470465<br/>
2: V14 27.048769<br/>
3: V17  6.237203<br/>
4: V10  4.884359<br/>
5: V20  3.527659<br/>
6:  V9  1.245999<br/>
7:  V4  1.071521<br/>
<br/>
Feature        Gain       Cover  Frequency<br/>
1:     V14 0.416218440 0.354986648 0.33766234<br/>
2:     V10 0.246135716 0.176385012 0.12337662<br/>
3:     V12 0.165018774 0.126950350 0.09090909<br/>
4:     V17 0.111386539 0.190137506 0.07142857<br/>
5:      V4 0.059261483 0.149853021 0.32467532<br/>
6:      V3 0.001180491 0.000687422 0.02597403<br/>
7:     V11 0.000798557 0.001000042 0.02597403<br/>

## About this Dataset<br/>

Context<br/>
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.<br/>

Content<br/>
The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.<br/>

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.<br/>

Inspiration<br/>
Identify fraudulent credit card transactions.<br/>

Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.<br/>

Acknowledgements<br/>
The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection. More details on current and past projects on related topics are available on https://www.researchgate.net/project/Fraud-detection-5 and the page of the DefeatFraud project<br/>

[Kaggle Reference](https://www.kaggle.com/samkirkiles/credit-card-fraud)
