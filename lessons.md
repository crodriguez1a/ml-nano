**Week of Dec 11, 2018**


| Project	Suggested Deadline  | | 
| --- | --- |
| Term Begins| 	Dec 11 |
| Predicting Boston Housing Prices	| Jan 1 |
| Finding Donors for CharityML	| Jan 22 |
| Creating Customer Segments	| Feb 5 |
| Term End Deadline	| Feb 26 |

| W1 | | |
| --- | --- | --- |
| Time | Activity | Notes |
| ~~1hr~~ | Complete Lesson 1: Welcome to ML | Prerequisites are important, but you can also learn them as you go.
| ~~3hr~~ | Complete Lesson 2: What is ML | These are just for overview, we will cover them again in future lessons.
| ~~2hr~~ q | Finish Practice Project: Titanic Survival Exploration Project | Here the goal is to get familiarized with all the libraries involved.

Lesson 2:

# Decision Trees

Qualifier for selecting features at the top of the tree - which one seems more decisive for splitting the data?

Deterministic, successive, traversing

# Naive Bayes

Example: Spam Classifier

- Manual classification
- Feature selection

Calculating probability (20/25 with the word "cheap" were classified as spam)

```
probability = events/number of outcomes
```

Utilize features associated with highest probability

# Gradient Descent

Smallest number of steps in the correct direction to arrive at a solution

# Linear Regression

Finding the line for a given dataset that best fits the trajectory of the data points.
Drawing a random line as starting point and Calculating error, then minimize error incrementally (aka, descending).

(Least Squares)
Since we don't want to measure negative distances from our line to our points, we use the squares.
https://en.wikipedia.org/wiki/Least_squares

# Logistic Regression

Finding a line that bests cuts/separates/classifies the data points

Starting with a random line, determine the number of misclassified points (errors). With Gradient Descent we can move the line until errors are reduced and points are correctly classified.

Log-loss function, we don't want to minimize the number of errors but something that captures the number of errors. Assigns large penalties to misclassified points. Minimizing the error function.

# Support Vector Machines

Cutting data, finding the line that best cuts the data. Measuring distances from points to a cutting line. Minimum of all the distances from points to a line is larger than the other. Our goal is to make that Minimum as large as possible (Maximize the distance)

# Neural Networks

Multiple lines creating regions combined into nodes. We use the and operator to combine the outputs multiple nodes. We compare multiple boundaries.

# Kernal Method/Trick

Non linear, multi-dimensional(plane) or curved lines.
Curve trick - Function/equation, to separate points.
3 Dimension Trick - (x,y,xy) on x y z axis, separating points onto different planes.


# K-means clustering

Optimizing the location of the center of a cluster of data points. Useful for when we know how many clusters we want to end up with.

# Hierarchical clustering

When we don't know how many clusters we want to end up with, controlling cluster distance.




| W2 | | |
| --- | --- | --- |
| Time | Activity | Notes |
| ~~1hr~~ | Complete Lesson 4: Career Services Available to you | Setup your career portal and update it periodically. |
| ~~.5hr~~ | Complete Lesson 5 quiz: Numpy and Pandas Assessment | Practice the quizzes on your own computer or on a workspace |
| ~~2hr~~ | Complete Lesson 6: Training and Testing Models | Make a note of these steps, you would be following the same in next projects. Bonus: What is a cross validation set? |
| ~~1hr~~ | Complete Lesson 7: Evalutation Metrics | The diagram in the wikipedia of Precision and recall makes it easy to understand |
| ~~1hr~~ | Complete Lesson 8: Model Selection | This section is important for debugging when your model is not performing as expected. |


**Week of Dec 18, 2018**

# Numpy


Great tutorial here...http://cs231n.github.io/python-numpy-tutorial/#numpy

Numpy is the core library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays.

A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers. **The number of dimensions is the rank of the array;** **the shape of an array is a tuple of integers giving the size of the array along each dimension.**

```python
# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"
```

Note as with python lists, negative indices count from the end of the list.


# Pandas

Tutorial:
https://pandas.pydata.org/pandas-docs/stable/10min.html#min

Visualization:
https://www.kaggle.com/residentmario/univariate-plotting-with-pandas

# Sklearn

Choosing estimators:

https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

Parameters explained (example):

https://scikit-learn.org/stable/modules/svm.html#svm-classification


# Distribution

A univariate distribution refers to the distribution of a single random variable. Note that the above characteristics we saw of a normal distribution are for the distribution of one normal random variable, representing a univariate distribution.

On the other hand, a multivariate distribution refers to the probability distribution of a group of random variables. For example, a multivariate normal distribution is used to specify the probabilities of returns of a group of n stocks. This has relevance because the returns of different stocks in the group influence each other’s behaviour, that is, the behaviour of one random variable in the group is influenced by the behaviour of another variable.

# Evaulate, Validate, and Improve

## Statistics Refresher
https://statistics.laerd.com/statistical-guides/descriptive-inferential-statistics.php

**Descriptive Statistics**

Descriptive statistics is the term given to the analysis of data that helps describe, show or summarize data in a meaningful way such that, for example, patterns might emerge from the data. Descriptive statistics do not, however, allow us to make conclusions beyond the data we have analysed or reach conclusions regarding any hypotheses we might have made. They are simply a way to describe our data.

**Inferential Statistics**

We have seen that descriptive statistics provide information about our immediate group of data. For example, we could calculate the mean and standard deviation of the exam marks for the 100 students and this could provide valuable information about this group of 100 students. Any group of data like this, which includes all the data you are interested in, is called a population. A population can be small or large, as long as it includes all the data you are interested in. For example, if you were only interested in the exam marks of 100 students, the 100 students would represent your population. Descriptive statistics are applied to populations, and the properties of populations, like the mean or standard deviation, are called parameters as they represent the whole population (i.e., everybody you are interested in).


# Regression and Classification

- Regression returns a value (predicts trajectory)
- Classification returns a state (+ or -)


# Confusion Matrix

Table that stores these four values

|| Spam Folder | Inbox |
|---|---|---|---|
|Spam| True Positive | False Positive |
|Not Spam| True Negative | False Negative |

|| Guessed Positive | Guessed Negative |
|---|---|---|---|
|Positive| True Positive | False Positive |
|Negative| True Negative | False Negative |

###Type 1 and Type 2 Errors

Sometimes in the literature, you'll see False Positives and False Negatives as Type 1 and Type 2 errors. Here is the correspondence:

**Type 1 Error** (Error of the first kind, or False Positive): In the medical example, this is when we misdiagnose a healthy patient as sick.

**Type 2 Error** (Error of the second kind, or False Negative): In the medical example, this is when we misdiagnose a sick patient as healthy.

## Accuracy

Ration between the number of correctly classified points and the number of total points

True Positive + True Negative / Total = n%

`accuracy_score` in sklearn

Accuracy = correctly classified points/all points

Accuracy won't work data is skewed. This affects the efficacy of an accuracy score

## Precision and Recall

High Recall vs High Precision Models

### Precision

Out of all the points predicted to be positive[sick], how many were actually positive[sick]?

Precision = Correctly classified as positive / total classified as positive


### Recall

Out of the points labeled positive how many did we correctly classify as positive

Recall = Correctly classified as positive / total actual positive


## F1 Score

Harmonic Mean Average - 2 * xy / x + y where x = precision and y = recall
Closer to the smallest between precision and recall raising a flag if one is small

## Fβ (beta) Score

Prioritizes either recall or precision
The smaller the beta the more we care about precision
The larger the beta the more we care about recall

If β=0, then we get precision.
If β=∞, then we get recall.
For other values of β, if they are close to 0, we get something close to precision, if they are large numbers, then we get something close to recall, and if β=1, then we get the harmonic mean of precision and recall.

## ROC Curve

Perflict SPlit 1.0
Good SPlit 0.8
Bad or Random SPlit 0.5

True Positive Rate = True Positives/ All Positives
False Positive Rate = False Positives / All Negatives
Two extrems are binary 1,1 0,0
Perfect split 0,1

The closer your area under the ROC curve is to one, the better your model.

## Regression Metrics

### Mean Absolute Error
Measure the fit using absolute values of distance from the points to the line. Absolute value function is not differentiable. Not good with Gradient Descent.

### Mean Squared Error
More common. Add the squares of the distances between the points and the line.

### R2 Score
Compare a model to simplest possible model
Good models are closer to 1, closer to zero is basically a guess


# Model Selection

## Types of Errors

Underfitting is often due to oversimplifying the problem/model
Underfitting doesn't do well in a traning set "Error due to Bias" "High Bias"

Overfitting is often due to overcomplicating the problem/model
Model is too specific, does well training set but memorizes instead of learning charasterics. "Error due to Variance" "High Variance"

## Model Complexity Graph

Visualizes underfitting, good, and overfitting models
We go from linear to polynomial with rising degree

## Cross Validation Set

A set that isn't testing or training, but used to validate our model selection without using the testing set.

## K-Fold Cross Validation

Randomizing cross validation data set Try `KFold` in sklearn with random param set to True

The data set is divided into k subsets and each time, one of the k subsets is used as the test set and the other k-1 subsets are put together to form a training set. Then the average error across all k trials is computed. This helps prevent overfitting.

## Grid Search

Make a table with all the possibilities of parameters and hyperparemeters for any given model