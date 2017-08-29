###########################
#### MSBX 5410 ############
### Data Project ##########
### Authored By: #########
#### Tyler Cady ##########
#########################

# Import Libraries
library(dplyr)
library(ggplot2)
library(RColorBrewer)
# display.brewer.all() # Looks at the available colors 

# Import Data 
bcd <- read.csv("~/Documents/Masters Program/MSBX 5410 Fundamentals of DA/Data Project/BreastCancerData.csv",
                header = TRUE)
# Check out the data
View(bcd)
summary(bcd)
str(bcd)
# Look into missing cases
na <- rep(NA,32)
for(i in 1:32){
  na[i] <- sum(is.na(bcd[,i]))
}
cols1 <- brewer.pal(8, "Set1")

plot(na, main = "Missing Data by Variable", xlab = "Index of Variables", 
     ylab = "Missing Values", col = c(rep(cols1,4)), pch = 18, cex = 3.5) # No Missing Data

# Create a new data frame for redundancy in the analyses
BC <- bcd
# Coerce id into a factor as this interpretation makes the most sense
BC$id <- as.factor(bcd$id)
# Investigate the distribution of Maglignant and Benign tumor cells 
table(BC$diagnosis)
prop.table(table(BC$diagnosis))
# Diagnosis Distribution Plot
diag.colors <- brewer.pal(3, 'Set1') # Colors for diagnosis distribution plot
nBC <- BC$diagnosis # Making a new vector to input into the plot
levels(nBC) <- c('Benign', 'Malignant') # Changing the factor levels for the new vector
# Plot it 
plot(nBC, col = diag.colors, ylim = c(0,400), xlab = 'Diagnosis', 
     ylab = 'Observations', main = 'Distribution of the Diagnosis \nfor Breast Cancer Tumors') 
# Build a Correlation Plot
library(corrplot)
# Exclude the id variable 
BC.cor <- BC[,2:32]
# Numerically encode the diagnosis factor
BC.cor$diagnosis <- as.character(BC.cor$diagnosis)
for(i in 1:length(BC.cor$diagnosis)){
  ifelse(BC.cor$diagnosis[i] == "B", BC.cor$diagnosis[i] <- 0, BC.cor$diagnosis[i] <- 1)
}
BC.cor$diagnosis <- as.numeric(BC.cor$diagnosis)
# Create a correlation matrix
BC.cor <- cor(as.matrix(BC.cor))
# Corrplot !!! Need to resize !!!
corrplot(BC.cor, method = "pie", type = "full", diag = TRUE,
         tl.pos = "n", 
         title = "Correlation Matirix Visualization")

corrplot(BC.cor, method = "pie", type = "full", diag = TRUE,
         title = "Correlation Matirix Visualization")
# rm(BC.cor)

# Investigate outliers 
boxplot(BC[,3:32], tl.srt = 45)
# Subset to better understand the outliers in each class
d <- filter(BC, diagnosis == 'M')
db <- filter(BC, diagnosis == 'B')
par(mfrow = c(1,2))
boxplot(scale(db[,3:32]))
boxplot(scale(d[,3:32]))

# K-Means CLustering ##################################################
library(cluster) # Clustering Package
library(HSAUR) # Produce LaTeX Tables

# Normalize the Data, leaving off the first two columns (ID and diagnosis)
BC_Norm <- scale(BC[,3:32], center = TRUE, scale = TRUE)
# Run the K-Means Clustering Algorithm
BC_cluster2 <- kmeans(BC_Norm, centers = 2, iter.max = 1000) 
# Dissimilarity Matrix Calculation 
dissE <- daisy(BC_Norm)
# Square the Dissimilarity Matrix
dE2 <- dissE^2
# Compute/Extract Silhouette Information from Clustering
sk2 <- silhouette(BC_cluster2$cl, dE2)
# Plot it
plot(sk2)

##################################################################################
# High degree of multicollinearity, create principal components 
# 1) Develop principal components (We will not use this in the analysis)

pc.matrix <- bcd[,3:32]
pc <- prcomp(pc.matrix, scale. = TRUE, center = TRUE)
library(ggplot2)
summary(pc)
pc$rotation
####################################################################################

# Modeling
library(caret) # Classification and Regression Package
library(randomForest) # Standard random forest package
library(1e1071) # Support Vector Machine (SVM) package
library(neuralnet) # Neural Network Package
library(pROC) # ROC curve package

## RANDOM FOREST #########################################################
dat <- BC[,2:32]
# Partition data into training and test set
set.seed(7)
part <- createDataPartition(dat$diagnosis, p = 0.75, list = FALSE)
train <- dat[part,]
test <- dat[-part,]

# Use the randomForest Package 
rf.mod <- randomForest(diagnosis ~ ., data = train, ntree = 2500, mtry = 7)

summary(rf.mod)
print(rf.mod)
rf.mod$confusion # Accuracy: 0.9697674 

# Determine How Many Trees to Build in the Forest
plot(rf.mod) # It looks lke we should try 500 and maybe 2500 trees

# The 500 tree model is the best of the two
rf.mod <- randomForest(diagnosis ~ ., data = train, ntree = 500, mtry = 7)

# Predictions and the Confusion Matrix
RfBase_pred <- predict(rf.mod, newdata = test)
table(test$diagnosis, RfBase_pred)

# Compute the Accuracy Metric 
AccRfBase <- sum(diag(table(test$diagnosis, RfBase_pred))) / sum(table(test$diagnosis, RfBase_pred))
print(AccRfBase) # Accuracy:  0.9577465

# Variable Importance Plot
varImpPlot(rf.mod, main = "Base Random Forest Variable Importance Plot")

# Using the Caret Package to Train the Model
control <- trainControl(method="repeatedcv", number=10, repeats = 3)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(train))
tunegrid <- expand.grid(.mtry=mtry)
caret.rf <- train(diagnosis ~ ., data=train, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(caret.rf) 
# Accuracy: 0.9640458 
# Kappa: 0.9223004

# Grid Search for the Optimal mtry Parameter for the Random Forest
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
set.seed(seed)
tunegrid <- expand.grid(.mtry=c(1:30)) # Check every configuration for mtry
caret.rf.grid.search <- train(diagnosis ~., data=train, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(caret.rf.grid.search)
plot(caret.rf.grid.search) # mtry = 7 is the optimal value of the parameter, with an accuracy of 0.9655962 and kappa of 0.9257066.

# Continue Tuning the Algorithm we Want the Optimal Threshold/Cutoff Point. 
control1 <- trainControl(method="repeatedcv", number=10, repeats=3, classProbs = TRUE, summaryFunction = twoClassSummary)
seed <- 7
metric1 <- "ROC"
set.seed(seed)
mtry1 <- 7
tunegrid1 <- expand.grid(mtry = mtry1)
caret.rf.thresh <- train(diagnosis ~ ., data=train, method="rf", metric= "ROC", tuneGrid=tunegrid1, trControl=control1)
print(caret.rf.thresh)
# AUC: 0.9850351
# Sensitivity: 0.9789174
# Specificity: 0.9409722

# Grid Search for Optimal Amount of Trees to Grow
control2 <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid", classProbs = TRUE, summaryFunction = twoClassSummary)
tunegrid2 <- expand.grid(mtry=7)
modellist <- list()
for (ntree in c(500, 1000, 1500, 2000, 2500)) {
  set.seed(seed)
  fit <- train(diagnosis ~ ., data=train, method="rf", metric="ROC", tuneGrid=tunegrid2, trControl=control2, ntree=ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit
}

#Compare the Results to Determine the Optimal Amount of Trees to Grow
results <- resamples(modellist) 
summary(results) # Use 1500 trees
dotplot(results, main = "Optimal Tree Growth") # Use 1500 tress 
#Train final/tuned model
set.seed(seed)
tunegrid3 <- expand.grid(mtry=7)
final.rf <- train(diagnosis ~ ., data = train, method = 'rf', metric = 'ROC', maximize = TRUE, tuneGrid = tunegrid3, 
                  ntree = 1500, trControl = control1, importance = TRUE )
print(final.rf)
final.rf$results
# mtry: 7
# AUC: 0.9855951
# AUC SD: 0.01920476
# Sensitivity: 0.9801519
# Sensitivity SD: 0.02530849
# Specificity: 0.9368056
# Specificity SD: 0.05298878

# Cross Validate with the test data 
predictions.rf <- predict.train(final.rf, newdata = test)
table(test$diagnosis, predictions.rf) # Confusion matrix
prop.table(table(test$diagnosis, predictions.rf)) # Proportion table of confusion matrix

# Calculate the Accuracy Metric
AccRF <- sum(diag(table(test$diagnosis, predictions.rf))) / sum(table(test$diagnosis, predictions.rf))
print(AccRF) # Accuracy: 0.9507042

# ROC curve to determine AUC
library(caTools)
pred_prob_rf <- predict(final.rf, test, type = 'prob')
colAUC(pred_prob_rf, test$diagnosis, plotROC=TRUE) # AUC: 0.994276
abline(a=0, b=1, col = "light grey")

## SVM ########################################################################
set.seed(8)
part <- createDataPartition(dat$diagnosis, p = 0.75, list = FALSE)
trainSVM <- dat[part,]
testSVM <- dat[-part,]

# Train the model to find optimal Cost
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
tunegridsvm <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))
set.seed(7)
svm_Linear_Grid <- train(diagnosis ~., data = trainSVM, method = "svmLinear",
                         trControl=trctrl,
                         preProcess = c("center", "scale"),
                         tuneGrid = tunegridsvm,metric='Accuracy',
                         tuneLength = 10)
print(svm_Linear_Grid) # Optimal C: 0.25
plot(svm_Linear_Grid) # Optimal C: 0.25

# Continue tuning the algorithm we want the optimal threshold/cutoff point. 
control1 <- trainControl(method="repeatedcv", number=10, repeats=3, classProbs = TRUE, summaryFunction = twoClassSummary)
seed <- 7
metric1 <- "ROC"
set.seed(seed)
C1 <- 0.25
tunegrid1 <- expand.grid(C = C1)
caret.svm.thresh <- train(diagnosis ~ ., data=trainSVM, method="svmLinear", metric= "ROC", tuneGrid=tunegrid1, trControl=control1)
print(caret.svm.thresh)
# AUC: 0.99434
# Sensitivity: 0.9913105
# Specificity: 0.9393056

#Run the model
set.seed(seed)
tunegrid3 <- expand.grid(C=0.25)
final.svm <- train(diagnosis ~ ., data = trainSVM, method = 'svmLinear', metric = 'ROC', maximize = TRUE, tuneGrid = tunegrid3, 
                   gamma = 0, trControl = control1, importance = TRUE )
print(final.svm)
final.svm$results
# C: 0.25
# AUC: 0.99434
# AUC SD: 0.0100454
# Sensitivity: 0.9913105
# Sensitivity SD: 0.01602212
# Specificity: 0.9393056
# Specificity SD: 0.06693269

#Cross validation
predictions.svm <- predict.train(final.svm, newdata = testSVM)
table(testSVM$diagnosis, predictions.svm) # Confusion matrix
prop.table(table(testSVM$diagnosis, predictions.svm)) 
plot(predictions.svm)

# Obtain the Accuracy Metric 
AccSVM <- sum(diag(table(testSVM$diagnosis, predictions.svm))) / sum(table(testSVM$diagnosis, predictions.svm))
print(AccSVM) # Accuracy: 0.971831 

# Plot the ROC Curve
library(caTools)
pred_prob_svm <- predict(final.svm, testSVM, type = 'prob')
colAUC(pred_prob_svm, testSVM$diagnosis, plotROC=TRUE) # AUC 0.99788
abline(a=0, b=1, col = "light grey")

## Artificial Neural networks ################################################

# Partition Data into Test and Training Sets
set.seed(9)
part <- createDataPartition(dat$diagnosis, p = 0.75, list = FALSE)
trainNN <- dat[part,]
testNN <- dat[-part,]

# Perform a grid search to Determine the Optimal Parameters
trctrl <- trainControl(method = "repeatedcv", number = 10, returnResamp = "all")
tunegridnn <- expand.grid(.decay = seq(from = 0.001, to = 0.008, by = 0.001), .size = seq(from = 22, to = 23, by = 1))
set.seed(7)
nn_Grid <- train(diagnosis ~., data = trainNN, method = "nnet",
                 trControl=trctrl,
                 preProcess = c("center", "scale"),
                 tuneGrid = tunegridnn,
                 metric="Accuracy")
print(nn_Grid)
plot(nn_Grid) # Optimal hyperparameters are .size = 22, decay = 0.008. 

# Continue Tuning with optimal parameters and the ROC metric
control1 <- trainControl(method="repeatedcv", number=10, repeats=3, classProbs = TRUE, summaryFunction = twoClassSummary)
seed <- 7
metric1 <- "ROC"
set.seed(seed)
tunegrid1 <- expand.grid(decay = 0.008, size = 22)
caret.nn.thresh <- train(diagnosis ~ ., data=trainNN, method="nnet", 
                         metric= "ROC", tuneGrid=tunegrid1, trControl=control1)
print(caret.nn.thresh)
# AUC: 0.977301
# Sensitivity: 0.9627255
# Specificity: 0.8870833

# Final Model 
set.seed(seed)
tunegrid3 <- expand.grid(decay = 0.008, size = 22)
final.nn <- train(diagnosis ~ ., data = trainNN, method = 'nnet',
                  preProcess = c("center", "scale"),
                  metric = 'ROC', maximize = TRUE, tuneGrid = tunegrid3, 
                  trControl = control1, importance = TRUE )
print(final.nn)
final.nn$results
# Decay: 0.008
# Size: 22
# AUC: 0.9932516
# AUC SD: 0.01272005
# Sensitivity: 0.981339
# Sensitivity SD: 0.02552711
# Specificity: 0.9533333
# Specificity SD: 0.05557303

# Predictions and Confusion Matrix
predictions.nn <- predict.train(final.nn, newdata = testNN)
table(testNN$diagnosis, predictions.nn) # Confusion matrix
prop.table(table(testNN$diagnosis, predictions.nn)) 
plot(predictions.nn)

# Calculate the Accuracy Metric
AccNN <- sum(diag(table(testNN$diagnosis, predictions.nn))) / sum(table(testNN$diagnosis, predictions.nn))
print(AccNN) # Accuracy: 0.9788732

# Plot ROC Curve
library(caTools)
pred_nn_prob <- predict(final.nn, testNN, type = 'prob')
colAUC(pred_nn_prob, testNN$diagnosis, plotROC=TRUE) # AUC 0.99788
abline(a=0, b=1, col = "light grey")




















