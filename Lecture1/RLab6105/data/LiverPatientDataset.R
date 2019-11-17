install.packages("caret")
library(caret)

data(iris)

# rename the dataset
dataset <- iris
  
# define the filename
filename <- "LiverPatientDataset.csv"

# load the CSV file from the local directory
dataset <- read.csv(filename, header=FALSE)

# set the column names in the dataset
colnames(dataset) <- c("Age","TB","DB","Alkphos","Sgpt","Sgot","TP","ALB","A/G Ratio","Selector")

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$Species, p=0.80, list=FALSE)

# select 20% of the data for validation
validation <- dataset[-validation_index,]

# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]

## ============================================================================
## 3. Summarize dataset
## ============================================================================

## 3.1 Dimensions of Dataset

# dimensions of dataset
dim(dataset)

## 3.2 Types of Attributes

# list types for each attribute
sapply(dataset, class)

# take a peek at the first 5 rows of the data
head(dataset)
  
# list the levels for the class
levels(dataset$Species)

# summarize the class distribution
percentage <- prop.table(table(dataset$Species)) * 100
cbind(freq=table(dataset$Species), percentage=percentage)

# summarize attribute distributions
summary(dataset)

## ============================================================================
## 4. Visualize dataset
## ============================================================================

# split input and output
x <- dataset[,1:4]
y <- dataset[,5]

Given that the input variables are numeric, we can create 
box and whisker plots of each.

# boxplot for each attribute on one image
par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(x[,i], main=names(iris)[i])
}

# barplot for class breakdown
plot(y)

# scatterplot matrix
install.packages("ISLR")
install.packages("stringi")

library(ISLR); library(ggplot2); library(caret);
featurePlot(x=x, y=y)
featurePlot(x=x, y=y, plot="pairs")

# box and whisker plots for each attribute
featurePlot(x=x, y=y, plot="box")

# density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

## ============================================================================
## 4. Evaluate ML algorithms
## ============================================================================

5.1 Test Harness
We will 10-fold crossvalidation to estimate accuracy.

# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

5.2 Build Models

Let's evaluate 3 different algorithms:
  
Linear Discriminant Analysis (LDA)
k-Nearest Neighbors (kNN).
Random Forest (RF)

Let's build our three models:
  
# a) linear algorithms
install.packages("e1071")
library(e1071)
set.seed(7)
fit.lda <- train(Species~., data=dataset, method="lda", metric=metric, trControl=control)

# b) nonlinear algorithms
# kNN
set.seed(7)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)

# c) advanced algorithms
# Random Forest
set.seed(7)
fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=control)

5.3 Select Best Model

# summarize accuracy of models
results <- resamples(list(lda=fit.lda, knn=fit.knn, rf=fit.rf))
summary(results)

# compare accuracy of models
dotplot(results)

# summarize Best Model
print(fit.lda)

## ============================================================================
## 6. Make predictions
## ============================================================================

# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)
