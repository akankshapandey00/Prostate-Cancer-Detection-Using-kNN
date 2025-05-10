# Install Pacakage
install.packages("class", repos = "http://cran.us.r-project.org")
install.packages("gmodels",repos = "http://cran.us.r-project.org")
install.packages("caret", repos = "http://cran.us.r-project.org")
install.packages("tidyverse", repos = "http://cran.us.r-project.org")
```


#Importing CSV file
pcr <- read.csv("/Users/akanksha/Downloads/Prostate_Cancer.csv", stringsAsFactors = FALSE)
# To confirm that the data is structured with 569 examples and 32 features
str(pcr)


# Step 1 - Preparing and exploring the data
#dropping ID to prevent overfitting
pcr <- pcr[-1]

#to recode the diagnosis_result variable
table(pcr$diagnosis_result)
pcr$diagnosis_result <- factor(pcr$diagnosis_result, levels = c("B", "M"),labels = c("Benign", "Malignant"))
round(prop.table(table(pcr$diagnosis_result)) * 100, digits = 1)

# Normalizing numeric data
#normalize function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
normalize(c(1, 2, 3, 4, 5))
pcr_n <- as.data.frame(lapply(pcr[2:9], normalize))
summary(pcr_n$radius)

# Creating training and test data set
#splitting the pcr_n data frame into pcr_train and pcr_test
pcr_train <- pcr_n[1:65, ]
pcr_test <- pcr_n[66:100, ]

#creating Labels
pcr_train_labels <- pcr[1:65, 1]
pcr_test_labels <- pcr[66:100, 1]


# Step 2 – Training a model on the data:
library(class)
pcr_test_pred <- knn(train = pcr_train, test = pcr_test, cl = pcr_train_labels, k = 10)

#Step 3 – Evaluating model performance

library(gmodels)
CrossTable(x = pcr_test_labels, y = pcr_test_pred,
           prop.chisq = FALSE)


#Step 4 – improving model performance
pcr_z <- as.data.frame(scale(pcr[-1]))
normalize(c(1, 2, 3, 4, 5))
pcr_z <- as.data.frame(lapply(pcr[2:9], normalize))
summary(pcr_z$radius)
prc_train <- pcr_z[1:52,]
prc_test <- pcr_z[53:81,]
prc_train_labels <- pcr[1:52, 1]
prc_test_labels <- pcr[53:81, 1] 
prc_test_pred <- knn(train = prc_train, test = prc_test,cl = prc_train_labels, k=9)
CrossTable(x = pcr_test_labels, y = pcr_test_pred,
           prop.chisq = FALSE)


#Using kNN from the caret Package:
# Download dataset directly from URL 

data <- read.csv("/Users/akanksha/Downloads/Prostate_Cancer.csv", stringsAsFactors = FALSE)


library(caret)
library(ISLR)
library(dplyr)


set.seed(300)
#Spliting data as training and test set. Using createDataPartition() function from caret
partition <- createDataPartition(y = data$diagnosis_result,p = 0.75,list = FALSE)
training <- data[partition,]
testing <- data[-partition,]

#Checking distribution in original data and partitioned data
prop.table(table(training$diagnosis_result)) * 100
prop.table(table(testing$diagnosis_result)) * 100
prop.table(table(data$diagnosis_result)) * 100


# Preprocessing:

kNN requires variables to be normalized or scaled. caret provides facility to preprocess data. I am going to choose centring and scaling

trainX <- training[,names(training) != "Direction"]
preProcValues <- preProcess(x = trainX,method = c("center", "scale"))
preProcValues

# Training and Train Control - KNN 
#Using train() function in order to use KNN method to this Train data and Train Classes
set.seed(400)
ctrl <- trainControl(method="repeatedcv",repeats = 3) 
knnFit <- train(diagnosis_result ~ ., data = training, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)
knnFit

#Plotting yields Number of Neighbours Vs accuracy
plot(knnFit)

# Using predict() function to execute the prediction on the test dataset 
knnPredict <- predict(knnFit,newdata = testing )

# Confusion matrix to see accuracy value and other parameter values
confusionMatrix(knnPredict,as.factor(testing$diagnosis_result))
mean(knnPredict == testing$diagnosis_result)

# Verifying 2 class Summary Function - KNN - Training and Train Control
ctrl <- trainControl(method="repeatedcv",repeats = 3,classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit <- train(diagnosis_result ~ ., data = training, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)
knnFit

plot(knnFit, print.thres = 0.5, type="S")


#Get the confusion matrix to see accuracy value and other parameter values
knnPredict <- predict(knnFit,newdata = testing )
#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(knnPredict, as.factor(testing$diagnosis_result))
mean(knnPredict == testing$diagnosis_result)



