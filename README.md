# Gbm (v‘2.1.3’) vs XGBboost(v‘0.81.0.1’) comparison, Credit Card Fraud Detection Binary Classification<br/>
## Experiment Environment<br/>
1. 27 core, 71 processors (Intel(R) Xeon(R) CPU @ 2.10GHz)
2. 528g total memory

## Experiment Setup<br/>
[Download Data from Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
1. Total dataset = 284,807 x 31 with last column being binary outcome.
2. 70% training and 30% testing.
3. default gbm vs three variations of xgboost

## Example Results<br/>
### Pipelined Processing<br/>
![](img/flowchart.png)

### Time<br/>
gbm: 4 mins
xgboost: 74s-76s
Conclusion: xgboost is clear winner.

### AUC<br/>
gbm: 0.9598
xgboost: 0.9744-0.9748
Conclusion: xgboost is a little better.

### Feature Importance<br/>
Feature gbm_fi
1: V12 52.470465
2: V14 27.048769
3: V17  6.237203
4: V10  4.884359
5: V20  3.527659
6:  V9  1.245999
7:  V4  1.071521

Feature        Gain       Cover  Frequency
1:     V14 0.416218440 0.354986648 0.33766234
2:     V10 0.246135716 0.176385012 0.12337662
3:     V12 0.165018774 0.126950350 0.09090909
4:     V17 0.111386539 0.190137506 0.07142857
5:      V4 0.059261483 0.149853021 0.32467532
6:      V3 0.001180491 0.000687422 0.02597403
7:     V11 0.000798557 0.001000042 0.02597403
