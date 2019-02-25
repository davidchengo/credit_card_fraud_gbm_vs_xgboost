# 27 core, 71 processor, Intel(R) Xeon(R) CPU @ 2.10GHz
# 528g memory
# Script based on https://www.kaggle.com/nschneider/gbm-vs-xgboost-vs-lightgbm/comments
options(java.parameters = "-Xmx10g")
setwd("/home/wxc011/RProjectsGIT/CreditFraud")

# Boring Setup
# Libraries
library(pROC, quietly=TRUE)
library(microbenchmark, quietly=TRUE)
library(tictoc)

# Set seed so the train/test split is reproducible
set.seed(42)

# Read in the data and split it into train/test subsets
tic()
print("load data...")
credit.card.data = read.csv("input/creditcard.csv")
# 284,807 x 31

toc()
# 15.916 sec elapsed

train.test.split <- sample(2 # sample(x) generates a random permutation of the elements of x (or 1:x).
                           , nrow(credit.card.data) # a positive number, the number of items to choose from
                           , replace = TRUE #   cannot take a sample larger than the population when 'replace = FALSE'
                           , prob = c(0.7, 0.3))
toc()

sum(train.test.split==1)/nrow(credit.card.data)
# [1] 0.7000249

sum(train.test.split==2)/nrow(credit.card.data)
# [1] 0.2999751

train = credit.card.data[train.test.split == 1,]
test = credit.card.data[train.test.split == 2,]

# view data
head(train)
table(train$Class)
table(train$Class)/nrow(train)
# 0      1 
# 199038    334
# 0.99832474 0.00167526 
# 1 is fraud

head(test)
table(test$Class)
table(test$Class)/nrow(test)
# 0       1 
# 85277   158 
# 0.998150641 0.001849359 

# extremely unbalanced

# Feature Creation (not needed for this dataset)

# Modeling
# Assumptions:
#   The data will be placed into the their preferred data formats before calling the models.
#   Models will not be trained with cross-validation.
#   If possible, different number of cores will be used during the speed analysis. (future mod)

######################################## 
# GBM
# Training the GBM is slow enough,

library(gbm, quietly=TRUE)
library(parallel)
# Get the time to train the GBM model
tic()

system.time(
  gbm.model <- gbm(Class ~ . 
                   , distribution = "bernoulli" #logistic regression for 0-1 outcomes
                   , data = rbind(train, test) 
                   , n.trees = 500
                   , interaction.depth = 3 #The maximum depth of variable interactions. 1 implies an additive model, 2 implies a model with up to 2-way interactions, etc.
                   , n.minobsinnode = 100 #minimum number of observations in the trees terminal nodes.
                   , shrinkage = 0.01 #a shrinkage parameter applied to each tree in the expansion. Also known as the learning rate or step-size reduction.
                   , bag.fraction = 0.5 #the fraction of the training set observations randomly selected to propose the next tree in the expansion. This introduces randomnesses into the model fit. If bag.fraction<1 then running the same model twice will result in similar but different fits. gbm uses the R random number generator so set.seed can ensure that the model can be reconstructed.
                   , train.fraction = nrow(train) / (nrow(train) + nrow(test)) #The first train.fraction * nrows(data) observations are used to fit the gbm and the remainder are used for computing out-of-sample estimates of the loss function.
                   , n.cores = detectCores()

  )
)
# 235.666 sec elapsed

# Determine best iteration based on test data
# Estimates the optimal number of boosting iterations for a gbm object and optionally plots various performance measures
# indicate the method used to estimate the optimal number of boosting iterations. method="OOB" computes the out-of-bag estimate and method="test" uses the test (or validation) dataset to compute an out-of-sample estimate. method="cv" extracts the optimal number of iterations using cross-validation if gbm was called with cv.folds>1
best.iter = gbm.perf(gbm.model, method = "test")
# 500

# Get feature importance
gbm.feature.imp = summary(gbm.model, n.trees = best.iter)
# print all non-zero relative importance
gbm.feature.imp[gbm.feature.imp$rel.inf>0,]
# top 10 features
gbm.feature.imp[1:10,]
# var      rel.inf
# V12       V12 5.247047e+01
# V14       V14 2.704877e+01
# V17       V17 6.237203e+00
# V10       V10 4.884359e+00
# V20       V20 3.527659e+00
# V9         V9 1.245999e+00
# V4         V4 1.071521e+00
# V26       V26 6.739450e-01
# V16       V16 4.903874e-01
# V7         V7 4.234114e-01

# Plot and calculate AUC on test data
# Number of trees used in the prediction. n.trees may be a vector in which case predictions are returned for each iteration specified
gbm.test = predict(gbm.model, newdata = test, n.trees = best.iter)
# Build a ROC curve using pROC
auc.gbm = roc(test$Class, gbm.test, plot = FALSE, col = "red", auc = TRUE, ci = TRUE)
plot(auc.gbm,
     print.auc = TRUE,
     # auc.polygon = TRUE,
     # grid=c(0.1, 0.2),
     # grid.col = c("green", "red"),
     # max.auc.polygon = TRUE,
     auc.polygon.col = "skyblue",
     # print.thres = TRUE,
     print.auc.x = 0.3,
     print.auc.y = 0.2)
# Data: gbm.test in 85277 controls (test$Class 0) < 158 cases (test$Class 1).
# Area under the curve: 0.9598
toc()

# note:
# Model interpretability is critical to businesses. If you want to use high performance models (GLM, RF, GBM, Deep Learning, H2O, Keras, xgboost, etc), you need to learn how to explain them. 

######################################## 
# xgboost
library(xgboost, quietly=TRUE)
xgb.data.train <- xgb.DMatrix(as.matrix(train[, colnames(train) != "Class"]), label = train$Class)
xgb.data.test <- xgb.DMatrix(as.matrix(test[, colnames(test) != "Class"]), label = test$Class)

# Get the time to train the xgboost model
tic()
xgb.bench.speed = microbenchmark(
  xgb.model.speed <- xgb.train(data = xgb.data.train
                               , params = list(objective = "binary:logistic" #logistic regression for binary classification. Output probability.
                                               , eta = 0.1 #learning rate: scale the contribution of each tree by a factor of 0 < eta < 1
                                               , max.depth = 3 #maximum depth of a tree. Default: 6
                                               , min_child_weight = 100 #The larger, the more conservative the algorithm will be. Default: 1
                                               , subsample = 1 #It is advised to use this parameter with eta and increase nrounds. Default: 1
                                               , colsample_bytree = 1 #subsample ratio of columns when constructing each tree. Default: 1
                                               , nthread = 3
                                               , eval_metric = "auc" #Default: metric will be assigned according to objective(rmse for regression, and error for classification, mean average precision for ranking).
                               )
                               , watchlist = list(test = xgb.data.test) #named list of xgb.DMatrix datasets to use for evaluating model performance. 
                               , nrounds = 500 #max number of boosting iterations.
                               , early_stopping_rounds = 40 #If set to an integer k, training with a validation set will stop if the performance doesn't improve for k rounds.
                               , print_every_n = 20 #Print each n-th iteration evaluation messages when verbose>0.
  )
  , times = 5L
)
toc()
print(xgb.bench.speed)
# Unit: seconds
# expr
# xgb.model.speed <- xgb.train(data = xgb.data.train, params = list(objective = "binary:logistic",      eta = 0.1, max.depth = 3, min_child_weight = 100, subsample = 1,      colsample_bytree = 1, nthread = 3, eval_metric = "auc"),      watchlist = list(test = xgb.data.test), nrounds = 500, early_stopping_rounds = 40,      print_every_n = 20)
# min       lq     mean   median      uq      max neval
# 15.73398 15.76795 16.28949 16.51026 16.6106 16.82465     5

print(xgb.model.speed$best_score)
# test-auc 
# 0.974415 

# Make predictions on test set for ROC curve
xgb.test.speed = predict(xgb.model.speed
                         , newdata = as.matrix(test[, colnames(test) != "Class"])
                         , ntreelimit = xgb.model.speed$bestInd)
auc.xgb.speed = roc(test$Class, xgb.test.speed, plot = TRUE, col = "blue")
print(auc.xgb.speed)
# Area under the curve: 0.9744

######################################## 
# Train a deeper xgboost model to compare accuarcy.
tic()
xgb.bench.acc = microbenchmark(
  xgb.model.acc <- xgb.train(data = xgb.data.train
                             , params = list(objective = "binary:logistic"
                                             , eta = 0.1
                                             , max.depth = 7
                                             , min_child_weight = 100
                                             , subsample = 1
                                             , colsample_bytree = 1
                                             , nthread = 3
                                             , eval_metric = "auc"
                             )
                             , watchlist = list(test = xgb.data.test)
                             , nrounds = 500
                             , early_stopping_rounds = 40
                             , print_every_n = 20
  )
  , times = 5L
)
toc()
print(xgb.bench.acc)
# Unit: seconds
# expr
# xgb.model.acc <- xgb.train(data = xgb.data.train, params = list(objective = "binary:logistic",      eta = 0.1, max.depth = 7, min_child_weight = 100, subsample = 1,      colsample_bytree = 1, nthread = 3, eval_metric = "auc"),      watchlist = list(test = xgb.data.test), nrounds = 500, early_stopping_rounds = 40,      print_every_n = 20)
# min       lq     mean   median       uq      max neval
# 15.09776 15.32379 15.47706 15.42335 15.53328 16.00713     5

print(xgb.model.acc$best_score)
# test-auc 
# 0.974588
#Get feature importance
xgb.feature.imp = xgb.importance(model = xgb.model.acc)

# Make predictions on test set for ROC curve
xgb.test.acc = predict(xgb.model.acc
                       , newdata = as.matrix(test[, colnames(test) != "Class"])
                       , ntreelimit = xgb.model.acc$bestInd)
auc.xgb.acc = roc(test$Class, xgb.test.acc, plot = TRUE, col = "blue")
print(auc.xgb.acc)
# Area under the curve: 0.9746

######################################## 
# xgBoost with Histogram
xgb.bench.hist = microbenchmark(
  xgb.model.hist <- xgb.train(data = xgb.data.train
                              , params = list(objective = "binary:logistic"
                                              , eta = 0.1
                                              , max.depth = 7
                                              , min_child_weight = 100
                                              , subsample = 1
                                              , colsample_bytree = 1
                                              , nthread = 3
                                              , eval_metric = "auc"
                                              , tree_method = "hist"
                                              , grow_policy = "lossguide"
                              )
                              , watchlist = list(test = xgb.data.test)
                              , nrounds = 500
                              , early_stopping_rounds = 40
                              , print_every_n = 20
  )
  , times = 5L
)
print(xgb.bench.hist)
print(xgb.model.hist$bestScore)

#Get feature importance
xgb.feature.imp = xgb.importance(model = xgb.model.hist)

# Make predictions on test set for ROC curve
xgb.test.hist = predict(xgb.model.hist
                        , newdata = as.matrix(test[, colnames(test) != "Class"])
                        , ntreelimit = xgb.model.hist$bestInd)

auc.xgb.hist = roc(test$Class, xgb.test.hist, plot = TRUE, col = "blue", ci = TRUE)
plot(auc.xgb.hist,
     print.auc = TRUE,
     # auc.polygon = TRUE,
     # grid=c(0.1, 0.2),
     # grid.col = c("green", "red"),
     # max.auc.polygon = TRUE,
     auc.polygon.col = "skyblue",
     # print.thres = TRUE,
     print.auc.x = 0.3,
     print.auc.y = 0.2)
print(auc.xgb.hist)
# Area under the curve: 0.9748

######################################## 
# AUC comparison
print(auc.gbm$auc)
# Area under the curve: 0.9598
print(auc.xgb.speed$auc)
# Area under the curve: 0.9744
print(auc.xgb.acc$auc)
# Area under the curve: 0.9746
print(auc.xgb.hist$auc)
# Area under the curve: 0.9748

######################################## 
# Feature Importance
# The top features selected by all three models were very similar. Although, my understanding is that GBM is only based on frequency. The top 5 features were the same expect for GBM selecting v20 as an important feature. It is interesting that xgboost selects so few features.
rownames(gbm.feature.imp) <- seq(nrow(gbm.feature.imp))
head(gbm.feature.imp, 7)
head(xgb.feature.imp, 7)
