

install.packages("caret")
library(caret)

data(CO2)

# rename the dataset
dataset <- iris

You now have the iris data loaded in R and accessible via 
the dataset variable.

# define the filename
filename <- "iris.csv"

# load the CSV file from the local directory
dataset <- read.csv(filename, header=FALSE)

# set the column names in the dataset
colnames(dataset) <- c("Plant","Type","Treatment","conc","uptake")

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$Treatment, p=0.80, list=FALSE)

# select 20% of the data for validation
validation <- dataset[-validation_index,]

# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]

You now have training data in the dataset variable and a 
validation set we will use later in the validation variable.



# dimensions of dataset
dim(dataset)

Instances are also called "observations", or "datapoints", or "rows" of data.
Features are also called "dimensions", or "attributes", or "columns" of data.

# list types for each attribute
sapply(dataset, class)
You should see that all of the inputs are double and that 
the class value is a factor:
  
Sepal.Length Sepal.Width Petal.Length Petal.Width Species 
"numeric" "numeric" "numeric" "numeric" "factor"

# take a peek at the first 5 rows of the data
head(dataset)

## 3.4 Levels of the Class
The class variable is a factor. A factor is a class that 
has multiple class labels or levels. Lets look at the 
levels:
  
# list the levels for the class
levels(dataset$Treatment)

In the results we can see that the class has 3 different labels:
 
[1] "setosa" "versicolor" "virginica"
This is a multi-class or a multinomial classification problem. 
If there were two levels, it would be a binary classification problem.

## 3.5 Class Distribution
Lets now take a look at the number of instances (rows) that 
belong to each class. We can view this as an absolute count
and as a percentage.

# summarize the class distribution
percentage <- prop.table(table(dataset$Treatment)) * 100
cbind(freq=table(dataset$Treatment), percentage=percentage)

We can see that each class has the same number of instances
(40, or 33% of the dataset)

## 3.6 Statistical Summary
Now finally, we can take a look at a summary of each 
attribute.

This includes the mean, the min and max values as well as 
some percentiles (25th, 50th or media and 75th e.g. values 
at this points if we ordered all the values for an attribute).

# summarize attribute distributions
summary(dataset)

## ============================================================================
## 4. Visualize dataset
## ============================================================================

We now have a basic idea about the data. We need to extend 
that with some visualizations.
 
Univariate plots to better understand each attribute.
Multivariate plots to better understand the relationships 
between attributes.

## 4.1 Univariate Plots
We start with some univariate plots, that is, plots of each
individual variable.

It is helpful with visualization to have a way to refer to 
just the input attributes and just the output attributes. 
Lets set that up and call the inputs attributes x and the 
output attribute (or class) y.

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

This gives us a much clearer idea of the distribution of 
the input attributes:
  
  Box and Whisker Plots in R

We can also create a barplot of the Species class variable 
to get a graphical representation of the class distribution
(generally uninteresting in this case because theyre even).

# barplot for class breakdown
plot(y)

This confirms what we learned in the last section, that the
instances are evenly distributed across the three class:
  
  Bar Plot of Iris Flower Species

## 4.2 Multivariate Plots
Now we can look at the interactions between the variables.

First lets look at scatterplots of all pairs of attributes
and color the points by class. In addition, because the 
scatterplots show that points for each class are generally 
separate, we can draw ellipses around them.

 # scatterplot matrix
install.packages("ISLR")
install.packages("stringi")

library(ISLR); library(ggplot2); library(caret);
featurePlot(x=x, y=y)
featurePlot(x=x, y=y, plot="pairs")


We can see some clear relationships between the input 
attributes (trends) and between attributes and the class 
values (ellipses). That is also what your brain does to
learn to distinguish different classes in your life. Say,
the difference between a sabeertooth tiger and a yummy
yummy antelope. Very important for the survival of our
ancestors!
  
  Scatterplot Matrix of Iris Data in R

We can also look at box and whisker plots of each input 
variable again, but this time broken down into separate 
plots for each class. This can help to tease out obvious 
linear separations between the classes.

# box and whisker plots for each attribute
featurePlot(x=x, y=y, plot="box")

Box and Whisker Plot of Iris data by Class Value

Next we can get an idea of the distribution of each attribute, 
again like the box and whisker plots, broken down by class 
value. Sometimes histograms are good for this, but in this 
case we will use some probability density plots to give nice 
smooth lines for each distribution.

# density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

Like The boxplots, we can see the difference in distribution
of each attribute by class value. We can also see the 
Gaussian-like distribution (bell curve) of each attribute.


## ============================================================================
## 4. Evaluate ML algorithms
## ============================================================================

Set-up the test harness to use 10-fold cross validation.
Build 3 different models to predict species from flower 
measurements. Select the best model.

5.1 Test Harness
We will use 10-fold crossvalidation to estimate accuracy.

This will split our dataset into 10 parts, train in 9 and 
test on 1 and release for all combinations of train-test 
splits. We will also repeat the process 3 times for each 
algorithm with different splits of the data into 10 groups,
in an effort to get a more accurate estimate.

# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

We are using the metric of "Accuracy" to evaluate models. 
This is a ratio of the number of correctly predicted 
instances divided by the total number of instances in the 
dataset, multiplied by 100 to give a percentage 
(e.g. 95% accurate). 

5.2 Build Models
Lets evaluate 3 different algorithms:
  
Linear Discriminant Analysis (LDA)
k-Nearest Neighbors (kNN).
Random Forest (RF)

This is a good mixture of simple linear (LDA), 
nonlinear (kNN) and complex nonlinear methods (RF). 
We reset the random number seed before each run to ensure 
that the evaluation of each algorithm is performed using 
exactly the same data splits. It ensures the results are 
directly comparable.

Lets build our three models:
  
# a) linear algorithms
#lda
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

Caret does support the configuration and tuning of the 
configuration of each model, but we are not going to cover 
that here.

5.3 Select Best Model


We can report on the accuracy of each model by first 
creating a list of the created models and using the 
summary function.

# summarize accuracy of models
results <- resamples(list(lda=fit.lda, knn=fit.knn, rf=fit.rf))
summary(results)

We can see the accuracy of each classifier and also other 
metrics like Kappa:

We can also create a plot of the model evaluation results 
and compare the spread and the mean accuracy of each model. 
There is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times
(10 fold cross validation).

# compare accuracy of models
dotplot(results)
  
Comparison of Machine Learning Algorithms on Iris Dataset in R

The results for just the LDA model can be summarized.

# summarize Best Model
print(fit.lda)

This gives a nice summary of what was used to train the 
model and the mean and standard deviation (SD) accuracy 
achieved, specifically 97.5% accuracy +/- 4%
  
#predictions
We can run the LDA model directly on the validation set 
and summarize the results in a confusion matrix.

# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)

  
the kappa statistic is a measure of how closely the 
instances classified by the machine learning classifier 
matched the data labeled as ground truth, controlling for 
the accuracy of a random classifier as measured by the 
expected accuracy. The kappa statistic for one model is 
directly comparable to the kappa statistic for any other 
model used for the same classification task.

Read here below for more detail on the kappa statistic:
  
The Kappa statistic (or value) is a metric that compares an
Observed Accuracy with an Expected Accuracy (random chance). 
The kappa statistic is used not only to evaluate a single 
classifier, but also to evaluate classifiers amongst 
themselves. In addition, it takes into account random 
chance (agreement with a random classifier), which generally
means it is less misleading than simply using accuracy as 
a metric (an Observed Accuracy of 80% is a lot less 
impressive with an Expected Accuracy of 75% versus an 
Expected Accuracy of 50%). 

Computation of Observed Accuracy and Expected Accuracy is 
integral to comprehension of the kappa statistic, and is 
most easily illustrated through use of a confusion matrix. 
Lets begin with a simple confusion matrix from a simple 
binary classification of Cats and Dogs:
  
  Computation

Cats Dogs
Cats| 10 | 7  |
Dogs| 5  | 8  |
  
Assume that a model was built using supervised machine 
learning on labeled data. This doesn't always have to be 
the case; the kappa statistic is often used as a measure 
of reliability between two human raters. Regardless, 
columns correspond to one "rater" while rows correspond to 
another "rater". In supervised machine learning, one 
"rater" reflects ground truth (the actual values of each 
instance to be classified), obtained from labeled data, 
and the other "rater" is the machine learning classifier 
used to perform the classification. Ultimately it doesn't 
matter which is which to compute the kappa statistic, but 
for clarity's sake lets say that the columns reflect ground
truth and the rows reflect the machine learning classifier 
classifications.

From the confusion matrix we can see there are 30 instances
total (10 + 7 + 5 + 8 = 30). According to the first column 
15 were labeled as Cats (10 + 5 = 15), and according to the
second column 15 were labeled as Dogs (7 + 8 = 15). We can 
also see that the model classified 17 instances as Cats (10
+ 7 = 17) and 13 instances as Dogs (5 + 8 = 13).

Observed Accuracy is simply the number of instances that 
were classified correctly throughout the entire confusion 
matrix, i.e. the number of instances that were labeled as 
Cats via ground truth and then classified as Cats by the 
machine learning classifier, or labeled as Dogs via ground 
truth and then classified as Dogs by the machine learning 
classifier. To calculate Observed Accuracy, we simply add 
the number of instances that the machine learning classifier
agreed with the ground truth label, and divide by the total
number of instances. For this confusion matrix, this would 
be 0.6 ((10 + 8) / 30 = 0.6).

Before we get to the equation for the kappa statistic, one 
more value is needed: the Expected Accuracy. This value is 
defined as the accuracy that any random classifier would be 
expected to achieve based on the confusion matrix. 
The Expected Accuracy is directly related to the number of 
instances of each class (Cats and Dogs), along with the 
number of instances that the machine learning classifier 
agreed with the ground truth label. To calculate Expected 
Accuracy for our confusion matrix, first multiply the 
marginal frequency of Cats for one "rater" by the marginal 
frequency of Cats for the second "rater", and divide by the 
total number of instances. The marginal frequency for a 
certain class by a certain "rater" is just the sum of all 
instances the "rater" indicated were that class. In our 
case, 15 (10 + 5 = 15) instances were labeled as Cats 
according to ground truth, and 17 (10 + 7 = 17) instances 
were classified as Cats by the machine learning classifier. This results in a value of 8.5 (15 * 17 / 30 = 8.5). This is then done for the second class as well (and can be repeated for each additional class if there are more than 2). 15 (7 + 8 = 15) instances were labeled as Dogs according to ground truth, and 13 (8 + 5 = 13) instances were classified as Dogs by the machine learning classifier. This results in a value of 6.5 (15 * 13 / 30 = 6.5). The final step is to add all these values together, and finally divide again by the total number of instances, resulting in an Expected Accuracy of 0.5 ((8.5 + 6.5) / 30 = 0.5). In our example, the Expected Accuracy turned out to be 50%, as will always be the case when either "rater" classifies each class with the same frequency in a binary classification (both Cats and Dogs contained 15 instances according to ground truth labels in our confusion matrix).

The kappa statistic can then be calculated using both the 
Observed Accuracy (0.60) and the Expected Accuracy (0.50) 
and the formula:

Kappa = (observed accuracy - expected accuracy)/(1 - expected accuracy)
So, in our case, the kappa statistic equals: (0.60 - 0.50)/(1 - 0.50) = 0.20.

Reference: https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english


## ============================================================================
## 6. Make predictions
## ============================================================================

The LDA was the most accurate model. Now we want to get an 
idea of the accuracy of the model on our validation set.

This will give us an independent final check on the 
accuracy of the best model. It is valuable to keep a 
validation set just in case you made a slip during such 
as overfitting to the training set or a data leak. 
Both will result in an overly optimistic result.

We can run the LDA model directly on the validation set 
and summarize the results in a confusion matrix.

# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)

We can see that the accuracy is 100%. 
It was a small validation dataset (20%), but this result 
is within our expected margin of 97% +/-4% suggesting we 
may have an accurate and a reliably accurate model.

You Can Do Machine Learning in R!

You do not need to understand everything. (at least not right now) Your goal is to run through the tutorial end-to-end and get a result. You do not need to understand everything on the first pass. List down your questions as you go. Make heavy use of the ?FunctionName help syntax in R to learn about all of the functions that you're using.

You do not need to know how the algorithms work, for now.

You do not need to be an R programmer. The syntax of the R 
language can be confusing. Just like other languages, focus
on function calls (e.g. function()) and assignments 
(e.g. a <- "b"). We start our class with R to get you used  
to running algorithms on massive data (rows = observations,
columns = features), instead of on a single datapoint. After
your homework for next week, you'll be comfortable with this
and we shift to python!

You do not need to be a machine learning expert. 
You can learn about the benefits and limitations of 
various algorithms later!
  
Now, for the important part: your homework! Repeat this lab
with a different dataset. You are free to download your own,
or to use CRAN-built-in datasets, such as for example:
data(diamonds)

Remember, observations are simply data points along the 
rows. It's the events that you depend on to learn about the
world. Features are the variables along the columns, they
are the independent variables that describe the datapoint.

