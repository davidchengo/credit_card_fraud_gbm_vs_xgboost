# gbm vs XGBboost comparison, Credit Card Fraud Detection Binary Classification<br/>
## Experiment Environment<br/>
1. 27 core, 71 processors (Intel(R) Xeon(R) CPU @ 2.10GHz)
2. 528g total memory
3. Gbm (v‘2.1.3’) vs XGBboost(v‘0.81.0.1’) 

## Experiment Setup<br/>
[Download Data from Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
1. Total dataset = 284,807 x 31 with last column being binary outcome.
2. 70% training and 30% testing.
3. default gbm vs three variations of xgboost

## Example Results<br/>
### Pipelined Processing<br/>
![](img/flowchart.png)

### Time<br/>
gbm: 4 mins<br/>
xgboost: 74s-76s<br/>
Conclusion: xgboost is clear winner.<br/>

### AUC<br/>
gbm: 0.961<br/>
xgboost: 0.9744-0.9748<br/>
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
