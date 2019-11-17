library(caret)

data(CO2)

# rename the dataset
dataset <- CO2

#You now have the iris data loaded in R and accessible via the dataset variable.

# define the filename
#filename <- "CO2.csv"

# load the CSV file from the local directory
#dataset <- read.csv(filename, header=FALSE)

# set the column names in the dataset
colnames(dataset) <- c("Plant","Type","Treatment","conc","uptake")

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$Plant, p=0.80, list=FALSE)

# select 20% of the data for validation
validation <- dataset[-validation_index,]

# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]

# dimensions of dataset
dim(dataset)

# list types for each attribute
sapply(dataset, class)

# take a peek at the first 5 rows of the data
head(dataset)

# list the levels for the class
levels(dataset$Plant)

# summarize the class distribution
percentage <- prop.table(table(dataset$Plant)) * 100
cbind(freq=table(dataset$Plant), percentage=percentage)

# summarize attribute distributions
summary(dataset)

## ============================================================================
## 4. Visualize dataset
## ============================================================================

# split input and output
x <- dataset[,c(2,3,4,5)]
y <- dataset[,1]
plot(y)


# scatterplot matrix
#install.packages("ISLR")
#install.packages("stringi")

library(ISLR); library(ggplot2); library(caret);
featurePlot(x=x, y=y)
featurePlot(x=x, y=y, plot="pairs")

# box and whisker plots for each attribute
featurePlot(x=x, y=y, plot="box")


# density plots for each attribute by class value
#scales <- list(x=list(relation="free"), y=list(relation="free"))
#featurePlot(x=x, y=y, plot="density", scales=scales)

## ============================================================================
## 4. Evaluate ML algorithms
## ============================================================================

#We will use 10-fold crossvalidation to estimate accuracy.

# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=20)
metric <- "Accuracy"

# a) linear algorithms
#lda
library(e1071)
set.seed(7)
fit.lda <- train(Plant~., data=dataset, method="lda", metric=metric, trControl=control)


# b) nonlinear algorithms
# kNN
set.seed(7)
fit.knn <- train(Plant~., data=dataset, method="knn", metric=metric, trControl=control)

# c) SVMLinear2 algorithm 
set.seed(7)
fit.svnLinear2 <-train(Plant~., data=dataset, method="svmLinear2", metric=metric, trControl=control)

# d) Random Forest Algorithm
set.seed(7)
fit.rf <-train(Plant~., data=dataset, method="rf", metric=metric, trControl=control)

# e) Ada Boost Classification Algortihm
set.seed(7)
fit.adaboost <- train(Plant~., data=dataset, method="adaboost", metric=metric, trControl=control)

# summarize accuracy of models
results <- resamples(list(lda=fit.lda, knn=fit.knn,svnLinear2=fit.svnLinear2,rf=fit.rf,adaboost=fit.adaboost))
summary(results)


# compare accuracy of models
dotplot(results)

# summarize Best Model
print(fit.lda)
print(fit.adaboost)
print(fit.knn)
print(fit.svnLinear2)
print(fit.rf)


# estimate skill of LDA on the validation dataset
predictions <- predict(fit.adaboost, validation)
confusionMatrix(predictions, validation$Plant)
