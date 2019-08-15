**Week of Dec 11, 2018**

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

## R Score (Correlation Coefficient)
In statistics, the **correlation coefficient r** measures the strength and direction of a linear relationship between two variables on a scatterplot. The value of r is always between +1 and –1. Closer to 1 being a strong or near perfect linear relationship.

## F1 Score

Harmonic Mean Average - 2 * xy / x + y where x = precision and y = recall
Closer to the smallest between precision and recall raising a flag if one is small. Represents the overall accuracy

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
Underfitting doesn't do well in a training set "Error due to Bias" "High Bias"

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

train k times
Each time using a different bucket as our test set and the remaining points as our trainings. We then average the results.

## Grid Search

Make a table with all the possibilities of parameters and hyperparemeters for any given model

## Boston Home Prices

- Data gathering: https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names

# Linear Regression

- Classification asks Yes or No
- Regression asks How Much?
	- Drawing the best fitting line

- Fitting a line Through Data
	- Moving closer to each of the points
- Moving a Line (utilizing straight line equation)
	- slope and intercept (parameters) y = w1(x) + w2 
	- w1 is slope
	- w2 is intercept (vertical intersection)
- Absolute Trick (taking into account horizontal distance)
	-  (p,q) y = (w1 + p)x + (w2 + 1) as coeficient alpha
	-  add p param or horizontal distance
	-  Learning Rate: Alpha is the coeficient for minimal change
	-  Applying a learning rate to slope and intercept (alpha) to move the line gradually.
- Square Trick (absolute trick + vertical distance) or squared distance
	- Add q param or vertical distance as coeficient (q - q prime)
	- q prime is where q intersects the line
	- This builds upon the absolute trick
- Gradient Descent (Descrease an error function by walking along the negative of its gradient)
	- Calculates Error from random line
	- Moving the line decreasing error
	- Using the Error function (using the derivative or gradient) 
- Mean Absolute Error (error function)
	- y - y(hat)
	- Sum of all the errors(absolute values) divided by m(number of points)
- Mean Sqared Error
	- y - y(hat) squared
	- Average of series of squares 	
- Minimizing Error Functions
	- Gradient step is applied squared trick 
- Mean vs Total Error
	- How to decide? It doesn't matter because sum or total error is just a multiple of the mean squared error
- Mini-batch Gradient Descent
	- Batch vs Stochastic
	- Stochastic is applying squared or absolute trick at every point one-by-one
	- Batch is applying at every point at the same time 
	- Mini-batch splits data into small batches (for computational cost). Each batch is used to update weights.
- Absolute Error vs Squared Error
	- Average vs Sum
- Higher Dimensions
	- Fitting a plane
	- n-1 dimensional hyperplane (n dimensions) 
- Multiple Linear Regression
	- Predictor is an "independent variable" 
- Closed Form Solution
	- Calculating weights and values in a Matrix of n dimensions 
- Linear Regression Warnings
	- Linear Regression Works Best When the Data is Linear
	- Linear Regression is Sensitive to Outliers
		- Do outliers point to a new trajectory or potential a quadratic or polynomial curve? If not that outliers could potentially be ignored.  
- Polynomial Regression
	- Consider higher degree polynomials (apply more weights) 
- Regularization (make sure models don't overfit)
	-  Adding complexity of the model to error function
	-  Calculating combined error (e.g., from each coeficient in polynomial)
	-  L1 Regularization adds absolute value (coeficient) to the error 
		-  L1 regularization is useful for feature selection, as it tends to turn the less relevant weights into zero.
	-  Adds complexity to error to predict future error
	-  Includes Lambda tuning
	-  Tuning how much to punish complexity with parameter called lambda
	-  lambda is an apetite for complexity
	-  L1 vs L2 See chart
		- L1 (biggest difference)
			- Feature Selection will reduce noise (removes irelevant columns)
			- It tends to turn a lot of weights into zero, while leaving only the most important ones, thus helping in feature selection.

		- L2 is used when most columns are relevant

# Perceptron Algorithm

- Linear Boundaries
	- Has a linear equation produces a prediction as a score
	- bias is an additional param
	- y hat should resemble y as closely as possible
	- wsub1xsub1+wsub2xsub2 + bias = 0
- Higher Dimensions
	- n-dimensional space
	- Wx + b = 0 vector(W) will have n dimensions 
	- boundary is n-1 dimensional hyperplane
- Perceptrons
	- Building block of neural network
	- encoding of our equation into a graph
	- Nodes that represent the linear equation adding coeficient to bias for consistent notation
	- Second node adds a step function which translates score as binary result (or some other calculated result)
- Perceptrons as Logical Operators (and | or )
	- And true, true = true etc. Bools represented as 0,1
	- XOR operator = if anyone of the inputs is true, output true, any other combination ouputs false
- Perceptron Trick
	- Finding a line to correctly split data
	- a mathematical trick that modifies the equation of the line, so that it comes closer to a particular point.
	-  repositioning the line gradually (learning rate x inputs). We use the input data to bring the line closer to the point slowly.
- Perceptron Algorithm

# Decision Trees
- Can be used for multi-class or non-binary classification
- Recommending Apps Example
	- Base on previous data, recommend app to download 
	- Adding nodes with traversal based on the most apparent patterns 
	- Determining axes that would help establish thresholds for each nodes in the tree
- Entropy (from Physics)
	- Ability of particles to move around 
	- Solid (Low), Liquid (Medium), Gas (High)
	- How data points can be re-organized
	- Opposite is called Knowledge, How much do we know about a datasets homogeny. The more we know the higher the knowledge.
- Entropy Formula 1
	- High knowledge makes for easier prediction? 
- Entropy Formula 2
	- Probability of a group is the product of all individual probability 
- Entropy Formula 3
	- Rule: The logarithm of product is the sum of the logarithms
	- Definition of Entropy is the average of the logarithms
	
	```javascript
	entropy = -(p1)*Math.log2(p1) -(p2)*Math.log2(p2)-(p3)*Math.log2(p3)
	```
- Multiclass Entropy
	- Sum of all 
- Maximizing Information Gain
	- Information gain = change in entropy
	- Calculate entropy is data then children, difference between parent and average of children is information gain
	- Alogrithm will choose a decision node based on the highest information gain 
- Random Forests
	- problem with decision trees
		- too many decisions, overfitting  
		- too many speficic boundaries
	- Multiple (random) trees make predictions
	- Increasing the number of trees results in each decision tree learning some aspect of the training data thereby reducing the likelihood of overfitting the data.
- Hyperparameters
	- Maximum Depth
	- Minimum number of samples per leaf (to avoid having unbalanced nodes)
		- If it's an integer, it's the number of minimum samples in the leaf. If it's a float, it'll be considered as the minimum percentage of samples on each leaf.
	- Minimum number of samples per split (same as per leaf but applied on any split of a node)
	- Maximum number of features (to avoid overfitting, complex trees)
	- Large depth very often causes overfitting, since a tree that is too deep, can memorize the data. Small depth can result in a very simple model, which may cause underfitting.
Small minimum samples per leaf may result in leaves with very few samples, which results in the model memorizing the data, or in other words, overfitting. Large minimum samples may result in the tree not having enough flexibility to get built, and may result in underfitting.

# Naive Bayes
- Based on Conditional Probability 

- Guess the Person
	- Initial guess is called the prior
	- final guess after new info (condition) is called posterior
	- guess is based on highest probably after inferring
- Known and Inferred
	-  known: P(A) P(R|A) We know the probability of R given A
	-  Bayes theorem gives the inferred or the probablility a given r P(A|R)
- Guess the Person Now
	- Prior vs Posterior Probablities
	- formula of conditional probability
		- product of the probabilities
		- normalize probablities divide by the sum of n  
- Bayes Theorem
	- re-calculating the probability of an event based on additional knowns
	- Posterior Probabilities caclulated after we knew that R occurred 
- False Positives
	- rate error is larger than number of possibilities
- Bayesian Learning
	-  prior, posterior, normalization
- Naive Bayes Alogrithm
	- Naive Assumption
		- assume that probabilities are independent
	- Conditional Probability  
- Normalization - dividing each probability by the sum of both
- Building a Spam Classifier
- Project Spam Classifier
- In short, the Bayes theorem calculates the probability of a certain event happening(in our case, a message being spam) based on the joint probabilistic distributions of certain other events(in our case, the appearance of certain words in a message). 
-  It is composed of a prior(the probabilities that we are aware of or that is given to us) and the posterior(the probabilities we are looking to compute using the priors).

# Aside
- Accuracy, Precision, Recall

**Accuracy** measures how often the classifier makes the correct prediction. It’s the ratio of the number of correct predictions to the total number of predictions (the number of test data points).

**Precision** tells us what proportion of messages we classified as spam, actually were spam. It is a ratio of true positives(words classified as spam, and which are actually spam) to all positives(all words classified as spam, irrespective of whether that was the correct classification), in other words it is the ratio of

`[True Positives/(True Positives + False Positives)]`

**Recall(sensitivity)** tells us what proportion of messages that actually were spam were classified by us as spam. It is a ratio of true positives(words classified as spam, and which are actually spam) to all the words that were actually spam, in other words it is the ratio of

`[True Positives/(True Positives + False Negatives)]`

For classification problems that are skewed in their classification distributions like in our case, for example if we had a 100 text messages and only 2 were spam and the rest 98 weren't, accuracy by itself is not a very good metric. We could classify 90 messages as not spam(including the 2 that were spam but we classify them as not spam, hence they would be false negatives) and 10 as spam(all 10 false positives) and still get a reasonably good accuracy score. For such cases, precision and recall come in very handy. These two metrics can be combined to get the F1 score, which is weighted average of the precision and recall scores. This score can range from 0 to 1, with 1 being the best possible F1 score.

# Support Vector Machines

- Classification algorithm. Finds the best possible boundary, the one that maintains the largest distance from the points.

- Minimizing Distances
	- Maximing the margin
	- minimizing the error
- Error Function Intuition
	- include points in the margin as classification error
- Perceptron Algorithm
	- Error function will punish points according to distance to the main line (in the margin)
	- error is the absolute value of wx + b
	- gradient descent minimizes the error calculated in the error function to find the best fit 
- Classification Error
	- error increases linearly by one
	- adding the result of the error caluclated for each the points in the margin 
- Margin Error
	- margin = 2/norm of W (sum of the squares of the components)
	- error = |W|squared 
	- norm = square root of the sum of the squares of the vectors in the margin
	- norm of the vector w2 sqaured (same as l2 regularization)
- Margin Error Calculation
	- bias is the shift from origin
	- shifting back to the origin removes bias
	- simply doubling the distance from one marginal line to the main line 
- Error Function
	-  
- The C Parameter
	- Flexibility to decide how much error to allow
	- Large C focuses on correctly classifying, small margin
	- Small C allows some error, large margin
	- Hyperparameter requires Grid Search
- Polynomial Kernel 1
	- x squared = 4 (two linear polynomials)
	- Kernel Trick 
- Polynomial Kernel 2
	- Circular Boundary method
	- Higher dimensions method
	- polynomial function instead of linear
- Polynomial Kernel 3
	- x squared + y sqared = z
	- SVM draws the best separating plane
	- kernel - a set of functions that will come to help us determine a polynomial boundary
	- degree of the polynomial kernel is a hyperparameter that we can train
- RBF Kernel 1
	- Polynomial line to create dimension
	- convert to boundaries
	- radial basis functions 
	- 2x - 4y + 1z = -1
- RBF Kernel 2
	- higher dimension
	- paraboloid intersected by a circle 
- RBF Kernel 3
	- Gamma Hyperparam
	- large gama narrow curve
	- small gama wide curve (paraboloid in high dimentions)
	- Gaussian and Normal Distribution
	- gamma = 1/2sigma squared 
- SVMs in sklearn

# Ensemble Methods
- Bootstrap Aggregating
		- An average of the results 
	- Boosting 
		- A weighted combination/average of the results 
		- 
- Bagging
	- Weak Learners and Strong Learners
	- Train a weak learner on a subset of data
	- Never partition the data for subsets
	- Voting, 2 or more votes wins, tiebreakers
- AdaBoost
	- Algorithm discovered in 1996
	- 2nd learner focuses on 1st learners mistakes (punishing misclassification) 
- Weighting the Data
	- Minimizing the sum of weights of incorrectly classified points 
- Weighting the Models
	- 1
		- Weighted by success/accuracy 
	- 2
		- Truth Model - Large Positive Model 1
		- Liar Model - Large Negative Weight 0.5
		- 50/50 - No weight 0
	- 3 
		- weight = natural logarithm * accuracy/1-accuracy
		- weight = ln(#correct/#incorrect)
		- zero denominator = infinity (always listen)
		- zero numerator = -infinity (always do the opposite)
- Combining the Models
	- Weight the vote by the corresponding weight
	- Sum of negative and positive weights
	- Sum of weak learners 
- AdaBoost in Sklearn
	- 
- Resources

# Unsupervised Learning

- Unsupervised Learning
	- Dataset without labels
	- Finding clusters
	- Finding shapes or one dimensional line
	- dimensionality reduction
- Clustering Movies
	- no target labels given, but classification can still happen 
- How Many Clusters?
	- K-Means is the most used algo and happens in two steps
		- Assign
			- Which points are closer to the cluster center? Assign them accordingly.
		- Optimize  
			- Opt the total quadratic distance/length
			- Move center to the most appropriate center where total distance is minimized.
- Match Points with Clusters
	- Iteratively Assigning and Optimizing 

- K-Means Visualization
	- https://www.naftaliharris.com/blog/visualizing-k-means-clustering/ 
- Sklearn Challenges of k-means
	- number of clusters
	- usability
	- suitability 
	- n_clusters (most important)
	- max_iter (assign and optimize iter) 300 is good value. It will typically terminate before then.
	- no_init: different initilizations of the algo will produce unique clustering. K-means will provide the ensemble the of those clusters.
- Limitations of k-means
	- You have to figure out how many clusters to try
- Limitations of k-means
- Counterintuitive clusters
	- Not every visualization of clusters is intuitive. Cluster centers (assignment) can be somewhat counterinutive. 
	- Local Minimum
		- Initialization can change assignment
		- You could have a bad local minimum, separating line could be misplaced during initialization.

# Silhouette Score

- The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette ranges from −1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.


# Hierarchical and Density-based Clustering

- K-means consideration
	- Cases where you know the number of clusters
	- Distance to centroid as a definition works better with sphere or hyper-spherical
	- Does not work with crescent and other shapes, dense datasets
- Other Clustering methods
	- Hierarchical
	- Density Based (datasets with noise), (two crescent) 
- Hierarchical Clustering - single-link
	- Choosing smallest distance between two clusters and attaching
	- Looks at the closest point (distance)
	- Creates a dendogram
	- Requires number of clusters
	- Can produce elongated clusters or lump together to much of the dataset
- Single-link Clustering vs K-means
	- Elongated clusters
	- Two Crescent
	- Two Rings
	- sparse clusters
	- works the same as k-means with dense clusters
	- Dendograms can visualize n-dimensional (hyperdimensional) datasets
- Complete-link, average-link, Ward (agglomerative clustering in scikit)
	- Complete Link 
		- Same way as single link
		- Assumes each point is a cluster
		- Meastures distance between points
		- Looks at farthest distance between two points (opposite of single link)
		- Looks at minimal distance between clusters, then groups the clusters
	- Average Link
		- Looks at distance between every point and finds the average
	- Ward's Method (default)
		- Calculates distance looking for a central point(average points). Subtracting (minimizing) variance already in cluster in each merging step.   
- Hierarchical implementation
	- `cluster.AgglomerativeClustering(n_clusters, linkage='ward') `
	- Dendograms require scipy library
	- Advantages
		- visual representations are informative
		- potent when data contains real hierachichal data
		- sensitive to noise and outliers
		- computationally intensive (not great with high dimensions)
	- Applications
		- Human microbiome  

- **DBSCAN** (Density Based Spacial Clustering of Applications with Noise)
	- Not every point is part of a cluster (noise)
	- Epsilon - Search distance around points
	- MinPts - min points to form a cluster
	- Identifying Core Points, Border Points
	- Shapes it performs well with (does not require no clusters)
		- Two Crescents 
		- Two Rings
		- Dense clusters
	- Advantages
		- Don't need to specify no of clusters 
		- Flexibility in shapes and sizes of clusters
		- Can deal with outliers
	- Disadvantages
		- Border points that are reachable from two clusters
		- Faces difficulties finding clusters of varying densities (use HDBSCAN instead)
	- Applications
		- Network Traffic
		- Anomaly Detection 

# Gaussian Mixture Models

- GMM Clustering
	- Every point belongs to different clusters but at different levels of membership 
	- Works well with tiger strips
	- Turning data into knowledge
	- Assumes that each cluster follows a certain statistical distribution
	- Gaussian Distribution
		- test scores
		- height 
- Gaussian Distribution
	- Histogram helps plot distribution (bell curve)
	- Mean/Average
	- Standard Deviation 
- GMM Clustering in one dimension
	- Mixture of more than one gaussian distribution
- Gaussian Distribution in 2D
	- Example of two test scores per student (Mean and SD)
	- Multivariate Gaussian Distribution (more than one variable)
	- Concentric Circles(elipse) Visualization 
- GMM in 2D
	- Made up of 2 separate Gaussian Distributions 
- Expectation Maximization Algorithm
	- Initialize K Gausian Distributions
	- Soft-cluster data (expectation or e step)
	- Re-estimate the gaussian (maximization or m)
	- Evaluate log-likelihood for convergence
	- Repeat step 2 until converged 
	- Example
		- Step 1 requires finding mean and std deviation for each distributions by running a k-means
		- Step 2 Soft Clustering (probability density function)
			- Uses Z (latent or hidden variable)
		- Step 3 Maximization Step
			- New gaussian parameters with inputs from step 2
			- New Mean comes from calculating weighted average of cluster points
			- Variance with weight as coeficient
			- Repeat steps until convergence
		- Step 4 Log-likelihood
			- The higher the log-likelihood the higher the confidence of membership
			- Mixin Coeficient
			- Mean
			- Variance 
			- Until Convergence
		- Spherical Covariance
			- Covariance Matrix (elipsis, rotation) vs spherical 
		- Initialization is important (k-means)
- GMM Implementation
	- Mixture library
	- n_components(clusters)
	- fit and predict 
- GMM Examples & Applications
	- Advantages
		- Soft clustering (sample membership)
			- classifying documents
		- Cluster shape flexibility
	- Disadvantages
		- Sensitive to init
		- possible to converge to local optimum
		- slow convergence
	- Applications
	   - Sensor Data (accelerometer, velocity)
	   		- Office activity vs commuting
	   		- Bike vs Walk 
	   	- Astronomy (stars, pulsars)
	   	- Biometrics
	   		- Speaker Verification
	   		- Fingerprints
	   	- Computer Vision
	   		- Each Pixel becomes a GMM
	   		- Can remove foreground from background

	   	- Paper: Nonparametric discovery of human routines from sensor data [PDF]

		- Paper: Application of the Gaussian mixture model in pulsar astronomy [PDF]

		- Paper: Speaker Verification Using Adapted Gaussian Mixture Models [PDF]

		- Paper: Adaptive background mixture models for real-time tracking [PDF]

Video: https://www.youtube.com/watch?v=lLt9H6RFO6A

- Cluster Analysis Process
	- From data to knowledge
	- feature selection/extraction
		- reducing dimensions
		- Extraction is transforming using PCA  
	- Choose a clustering algorithm
		- Choose a proximity measure 
			- documents or word embeddings (use cosign distance)
	- Cluster Validation
		- visualize result
		- scoring method (index)
	- Results Interpretation
		- Insights (domain expertise)    
- Cluster Validation
	- Procedure of evaulating the results objectively and quantitavely
	- External indices (labeled data)
	- Internal Indices (un-labeled/un-supervised)
	- Relative Indices (which of two clustering methods is better)
	- Compactness, Separability  
- External Validation Indices
	- When we have labeled data
	- Provide a score [0,1] or [-1,1]
	- Adjusted Rand Index
		- Comparing clustering to labeled data
		- number of pairs in same vs different cluster		- original labels are called ground truth
		- calculates closeness to grounded truth
		- ARI does not care what label we assign a cluster, as long as the point assignment matches that of the ground truth.
- Internal Validation Indices
	- Unspervised 
- Silhouette Coefficient
	- score [-1,1]
	- Coefficient for each point in the dataset
		- a = average distance to other samples
		- b = average distance to samples in closest neighboring cluster
		- done for every point and finds the average
	- Can be used for finding K
		 - Can be used to validate visualized intuition
		 - Penalizing clusters without enough distance between them
		 - Silhouette coefficient
			- Used to compare clustering algorithms
			- not the index we should use for DBScan, no concept of noise. SC rewards dense compact clustering.
			- not built to carve out concentric circles of patterns
			- works better with density
		- Internal Index for density based DBCV
	   - Since we can calculate the silhouette coefficient for each point, we can just average them across a cluster or an entire dataset.
- Lab

# Feature Scaling

- Re-scaling values to represent values between 0,1 (percentage)
- pros - reliable number in ouput
- cons - outliers may skew ouput, extreme values
- SVM, and K-means require scaling
	- Distance calculation trades off one dimension against the other
	- Anything measuring distance in 2D

# PCA - Principal Component Analysis

- Feature set compression
- One dimensional vs n dimensional data
- Data where only one axis (feature) provides information can considered one-dimensional
- PCA specializes on shift and rotation from the coordinates system
- PCA finds a new coordinate system that centers itself along the data, moves or rotates the x axis, and further axes as well
- Returns Spread value for these axes, eigenvalue, importance vector, how important to take each vector
- Dominant Axes - Major axis dominates" means that there's much more variation in one dimension than the other.
- Measurable vs Latent Features
	- Latent Variables - can't be measured directly but driven the phenomenon that you're measuring behind the scenes. Not exactly measurable with a single number.
	- Measurable features can be assigned a value (number) directly. 
	- Feature Selection
		- SelectKBest - when you can infer the number of latent features that will be important. Providing that number as K will output the K number of features with the highest K score
		- SelectPercentile - outputs the top features based on there score's percentage. 
	- Preserving Information (Compression)
		- Composite Features
		- PCA to reduce dimensionality
		- powerful independently for unsupervised learning
		- Projection projects principal component to one dimension
		- Variance - willingness or flexibility of an algorith to learn
		- Statistical Viariance - roughly the "spread" of a data distribution (similar to standard deviation)
		- PCA is determine by the direction of maximum variance (direction with longest variance)
	- Maximal Variance and Information Loss
		- Information loss is the distance between the points and their projected positions on the principal components
		- Maximizing variance will minimize distance from the old point to the new transformed point or Minimizing the Information Loss
	- Second PCA Strategy (unsupervised)
		- combining features (picking out latent features)
		- powerful unsupervised learning technique for when no domain knowledge is available 
	- Review/Definition of PCA
		- systemized way to transform input features into principal components
		- use PCs as new features
		- PCs are directions in data that maximaize variance (minimize loss) when you project/compress down on to them
		- more variance of data along a PC, higher that PC is ranked
		- most variance/most information -> first PC
		- second-most variance (most overlapping w/first PC) -> second pc
		- Number of PCs = no of input features
	- When to Use PCA
		- latent features driving the patterns in the data (surface a latent feature)
		- dimensionality reduction
			- visualize high dimensional data
				- project high dimensions to 2d scatter plot
			- reduce noise, throwing away less important PCs
			- Preprocessing for other algorithms who would benefit from fewer inputs
			
	- PCA for Facial Recognition
 		- eigenfaces - reducing dimensions for facial recognition
 	- Explained Variance Ratio
 		- The first component has x % variation of the data, second component has x % of the data, and so on.

# Random Projection and ICA

- Random Projection
	- More computationally efficient that PCA
	- Good for more very large dimensions
	- Chooses a random line instead of a line that represents maximize variance (minimize loss) (eg.,pca).
	- Simply multiplies by a random matrix
	- From d dimensions to k dimensions
	- Based on the Johnson-Lindenstrauss lemma
		- n points in high dimensional Euclideanspace can mapped down to a space in much lower dimensions preserving the distance between points to a large degree.
	- epsilon is our threshold for error related to distance between points 
- SparseRandomProject is little more performant that gaussian
- Independent Component Analysis (ICA)
	- Similar to PCA and RP, except that it assumes that features are mixtures of independent sources
	- Cocktail Party Problem
		- With a three dimensional dataset, known number of components expected.
- FastICA Algorithm
	- Approximating or finding the best W (weight for each component)
	- Helsinki University Paper
	- Sklearn implementation steps
		1. X, center and whiten
		1. intial random weight (W)
		1. W contains vectors
		1. decorrelate W, preventing convergence
		1. repeat step 3 until convergence 
	- Assumptions
		- Components should be statistically independent
		- Non Gaussian distributions
			- Distribution of a sum of independent variables tends towards a Gaussian distribution
			- W maximizes non-gaussianity
		- Quiz
			- ICA needs as many observations as the original signals we are trying to separate. 	
- ICA Lab
	- FastICA
	- X can be a zipped list of arrays 
- ICA Applications
	- Medical Scanners
		- EEG, MEG
		- FMRI
	- Financial Data
		- Factor model in finance
			- Stock signals
			- What caused stocks to rise and fall
		-  Cash Flow of 5 stores Time Series
			- First component captured spike during Christmas time    

- From the unsupervised learning assessment
	- Note to not confuse K-nearest neighbors and K-means clustering. K-nearest neighbors is a classification algorithm, which is a subset of supervised learning. K-means is a clustering algorithm, which is a subset of unsupervised learning. 
	- Note to not confuse K-nearest neighbors and K-means clustering. K-nearest neighbors is a classification algorithm, which is a subset of supervised learning. K-means is a clustering algorithm, which is a subset of unsupervised learning. 
	- When evaluating how many components to include in PCA, what is a good rule of thumb for the total amount of variance to be captured by the kept components? 80%


# Deep Learning

## Deep Neural Networks

- DNNs can find a more complex line that separates data points as opposed to normal neural networks

- **Non Linear Regression**

	- Redefine perceptron algorithm to work with other shapes

- **Error Function**
	- Gradient Descent - shortest path to minimum error
	- Many times a local minimum will help solve our problem
	- Discrete vs Continuous Gradient Descent
   		- Our error function cannot be discrete and should be continuous
	 	- Aztec pyramid vs Mount Errorist

- **Log-loss Error Function**
		
	- Assigning penalties to misclassified points
	- Misclassified points will add larger values to error sum
	- Reducing error to minimum possible value
	- Conditions to apply Gradient Descent
		- error function should be differentiable
		- error function should be continuous   

- **Discrete vs Continuous**
	
	- Moving from discrete predictions to continuous 
	- Discrete answers are yes/no
	- Continuous answers will be a number usually between 0 and 1
		- probability is a function of the distance of the line
		- Continuous predictions use sigmoid function whereas discrete predictions use step functions as its activation function
		- Sigmoid function
			- for large positive numbers will give values close to 1
			- for large negative numbers will give values close to 0
			- values close to 0 will give you values close 0.5

- **Multi-Class Classification and Softmax**
	- Dog, Cat, Bird
	- Softmax is the equivalent of the sigmoid activation function
	- Scaling class score using the exponential function to translage scores as probability
- **One-Hot Encoding** 
	-  One variable for each of the classes set to a binary value
- **Maximum Likelihood**
	- We pick the model that gives the existing labels the highest probability
	- probablity of i = yhat = signmoid (wx+b)
	- calculates the product of the probababilities for all classes
	- Maximize probability
- **Maximizing Probabilities**
	- Minimizing Error
	- Convert products into sums with log `log(ab) = log(a) + log(b)` 
- **Cross-Entropy 1**
	- Sum of the natural log of each probability
	- take the negative of logarithm of the probabilities or cross-entropy
	- pair each logarithm with the point where it came from, we get error
	- Minimizing cross entropy
	- cross-entropy is the connection between probability and error
- **Cross-Entropy 2**
	- High likelihood means small cross-entropy
	- Small likelihood means large cross-entropy
	- cross-entropy tells us when two vectors are similar or different
- **Multi-Class Cross Entropy**
	- negative of the summation of y - n and J - m 
- **Logistic Regression**
	- multi-class, we use cross entropy
	- binary, we use error function 
- **Gradient Descent**
	- Error function is a function of the weights
	- Gradient is the vector formed by the derivatives of the function 
	- If a point is well classified, we will get a small gradient. And if it's poorly classified, the gradient will be quite large.
- **Logistic Regression Algorithm**
	- Similar to Perceptorn algorithm
	- assigns weights to individual points to pull the line closer to those points 
- **Pre-Lab: Gradient Descent**
	-  
- **Notebook: Gradient Descent**
- Perceptron vs Gradient Descent
	- Perceptron doesn't change weights for correctly classified points, while Gradient descent takes steps in either direction changing weights for both correctly and incorrectly classified points. 

# Deep Neural Networks

- Non-linear Data

- Continuous Perceptrons
	- Perceptron returns the probability of a point being classified

- Non-Linear Models
	- Not separable with a line
	- Create a probability function with a curve that represents a set of points that are equally likely to be "blue" or "red"

- Neural Network Architecture
	- Combine two linear models into a non linear model
	- Almost like doing arithmetic on two models
	- To combine, we apply sigmoid function to scale sum of two points from each linear model to create a non-linear combination.
	- You can use weight and bias with each linear model when combining.
	- Combining NNs uses a similar approach to Perceptrons 
	- Layers
		- Input Layer
		- Hidden (Combination) Layer
		- Output Layer
	- Multi class simply uses multiple outputs
		- Use softmax to obtain well-defined probabilities 

- Feedfoward 
	- The process of continously applying combinations (matrix multiplications) to an input vector  using sigmoid until reach the desired output.
	- Training neural networks
		- Define the error function
		- A measure of the error of each point 

- Backpropagation
	- Running the feedforward operation backwards (backpropagation) to spread the error to each of the weights.
	- Use this to update the weights, and get a better model.
	- Continue this until we have a model that is good.
	
## Training Optimization

- Early Stopping Algorithm
	- We run Gradient Descent until training error stops decreasing. That's how we determine the number of epochs. At the point where increasing begins, we stop.

- Regularization
	- applying sigmoid to small values, we get a shallow slope, whereas larger values create a steeper slope (derivatives are more extreme) and give less room for gradient descent.
	- To prevent overfitting, we want to penalize high coeficients (utilizing lambda)
		- L1 sum of absolute values of the weights
			- Good for feature selection
			- Sparse vectors, small weights go to zero in turn minimizing the set 
		- L2 sum of squares of the weights
			- tries to maintain all waits homogeneously small
			- used the most
			- better for training models 

- Dropout
	- One part of the network with large weights dominates all of the training
	- to solve, we'll turn off parts of the network and let other nodes train
	- In a particular epoch we may not want to use certain nodes
	- We do this by calculating the probability that each node gets dropped at a particular epoch 
	
- Local Minima
	- Not being able to descend any further at a particular point in the curve

- Vanishing Gradient
	- Derivative tells us in what direction to move so if it is close to zero, it becomes unhelpful and training difficult making steps to small
	- Both sigmoid and tanh are susceptible to vanishing gradient
- Other Activation Functions
	- Fixing Vanishing Gradient and Local Minima
	- Hyperbolic Tangent Function (larger derivatives)
	- Rectified Linear Unit (ReLU) maximum between x and 0. Derivative is always one if number is positive. Binary. Also increases derivative size.

- Batch vs Stochastic Gradient Descent
	- Number of steps = number of epochs
	- in each epoch, all of the data is run through the network
	- calculate error, and backpropogate the error
	- this is done for all the data
	- if data is large, these epochs become huge matrix computations
	- we don't need to use all the data for each step/epoch
	- with Stochastic, we create a window (subset of data) 
	- We can split the data into batches and run them through, calculate error and their gradients and them backpropogate, creating an better boundary region. More steps are less accurate, but that's okay during training.

- Learning Rate Decay
	- If your learning rate is too big, you're taking huge steps, you may miss the minimum and keep going.
	- Decrease rate for smaller steps are more accurate and will arrive at local minimum 

- Random Restart
	- Start from a few different random places and run gradient descent from all of them arrive at a pretty good local minimum

- Momentum
	- solving the local minimum problem
	- taking the average of the last few steps when derivative becomes close to zero
	- requires weighting the steps so that more recent steps are more relevant
	- Once we get to the global minimum, we may overshoot but not by much

- Optimizers in Keras

	**SGD**
	This is Stochastic Gradient Descent. It uses the following parameters:
	
	Learning rate.
	Momentum (This takes the weighted average of the previous steps, in order to get a bit of momentum and go over bumps, as a way to not get stuck in local minima).
	Nesterov Momentum (This slows down the gradient when it's close to the solution).
	
	**Adam**
	Adam (Adaptive Moment Estimation) uses a more complicated exponential decay that consists of not just considering the average (first moment), but also the variance (second moment) of the previous steps.
	
	**RMSProp**
	RMSProp (RMS stands for Root Mean Squared Error) decreases the learning rate by dividing it by an exponentially decaying average of squared gradients.   	 
- Neural Network Regressions
	- Replacing the output sigmoid function with a an error function for classification
	- Combining linear functions with a relu, or sigmoid or tanh
	- We can use networks for regression just by removing the final activation function (no probability) 

Reference: https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/

# Convolutional Neural Networks

- Alexis Cook

- Applications of CNNs
	- NLP
		- Recurrent NN used more frequently 
	- Computer Vision
		- Teaching agents to play video games 
	- Voice User Interfaces
		- Google WaveNet, text to voice  
	- Sentiment
	
- How Computers Interpret Images
	- MNIST Database
		- 70,000 images of hand-written digits
	- Pixels are interpreted as a matrix
		- 255 for white
		- 0 for black
		- preprocessing - images are rescaled, labels are one-hot encoded (using vector index)
		- We flatten our matrix to a vector
		- feed to input layer of an MLP (multi-layer perceptron) in keras
		
- MLPs for Image Classification
	- two hidden layers with the same number of nodes, with relu activation
	- relu interprets values as binary
	- Flatten is an api in Keras

- Categorical Cross-Entropy (loss function)
	- Good for multi-class
	- compares models predection to true label (one-hot)
	- returns a higher value for the loss
	- Loss is lower when a label and prediction agree
	- Loss is higher when label and prediction disagree
	- Try to find params that minimize the loss function
	- Find minimum of the loss using an implementation of Gradient Descent as an optimizer
		- Stochastic Gradient Descent
		- Momentum
		- Adagrad
		- Adadelta
		- Rmsprop 

- Model Validation in Keras
	- Its not always clear how many layers to use, how many nodes to includes, or how many epochs
	- Good to split into Training, Validation, Test sets
	- Saving weights from each potential architecture for comparison using ModelCheckpoint class, save_best_only=True for only the most accurate weights, you can pass the checkpoint as a param to the fit method
	- fit method accepts split as a param

- When do MLPs (not) work well?
	- MLP requires deterministic spacial awareness
	- CNN understands spacial data based on proximity 

- Mini Project
	- Overfitting is detected by comparing the validation loss to the training loss. If the training loss is much lower than the validation loss, then the model might be overfitting.
	- Deep learning is not well-understood, and the practice is ahead of the theory in many cases. If you are new to deep learning, you are strongly encouraged to experiment with many models, to develop intuition about why models work.
	- **Tuning**
		- [x] Increase (or decrease) the number of nodes in each of the hidden layers.  Do you notice evidence of overfitting (or underfitting)?
		- [x] Increase (or decrease) the number of hidden layers.  Do you notice evidence of overfitting (or underfitting)?
		- [x] Remove the dropout layers in the network.  Do you notice evidence of overfitting?
	   - [x] Remove the ReLU activation functions.  Does the accuracy decrease?
		- [x] Remove the image pre-processing step with dividing every pixel by 255.  Does the accuracy decrease?
		- [x] Try a different optimizer, such as stochastic gradient descent.
		- [ ] Increase (or decrease) the batch size. 

- Local Connectivity
	- Techniques for complex images or datasets
	- MLPs use a lot of parameters
	- Only use fully connected layers
	- Only accept vectors as input
		- Throwing away all 2d information in favor of flattening
	- CNNs
		- connections between layers are informed by sparsely connected layers
		- accepts matrix as input   
		- hidden nodes only need to be connected to relevant parts of an image (regional breakdown)
		- Less prone to overfitting
		- Truly understand how to tease out patterns contained in image data
		- Selectively and conservatively add weights to the model
		- each of the hidden nodes can share a common set of weights
		- every pattern that is relevant to understand ing the image can appear anywhere (eg. high res)
		- positioning or placement of an image in a window shouldn't matter
- Convolutional Layers 1
	- Uses locally connected layers to inform convolutional layers
	- First select a width and height that define the convolution window
	- slide the window over the matrix and define a collection of pixels to which we connect a hidden node (in a convolutional layer
	- sum of product of weights for each node (with bias)
	- convolutional layers always receive relu activation function
	- weights are represented in a grid (aka Filter)
	- multiply each input node with corresponding weights, then apply relu
	- can interpret patterns with visualizing the filters
	- filter is a window of weights applied to a region of the image 
	- Decreasing the number of filters in a convolutionaly layer can help reduce overfitting
- Convolutional Layers 2
	- Filters are needed for each type of image charasteristics
	- Common to have tens to hundreds of window collections each corresponding to their own filter
	- filters are convolved across the height and width of image to produce an entire collections of nodes inside the convolutions layer
	- each filter in the collection is called an activation map
	- collectives are called feature maps
	- when visualizing, we can see filtered images
	- these filters discovering vertical or horizontal edges (edge detectors)
	- greyscale are interpreted as a 2d array (width, height)
	- color are interpeted as 3d array (width, height, depth)
	- to peform convolution on a color image, filter becomes three dimensional as well
	- Same calculation as before but with more dimensions
	- each 3d filter is really just a stack of 3 2d layers
	- filters can be fed as inputs to new layers
	- Dense layers are fully connected whereas convolutional layers are only connected to a small subset of the previous layers
	- both use inference the same way, weights and bias are intially randomly generated, thusly the filters and patters also
	- always specify a loss function
	- multi-class -> categorical crossentropy loss
	- cnn determines what patterns in needs to detect depending on the loss function
	- with cnns we don't specify the values of filters or what kind of patterns to detect
- Stride and Padding
	- Control the behavior of a convolutional layer by specifying the number of filters and the size of each filter
	- to increase the number of nodes increase the number of filters
	- increase size of detected patterns increase filters
	- Stride is the amount to which the filter moves over the image
		- reduce the window size for filter to move
	- dealing with nodes where filter extends beyond the image
		- you can pad with zeros to have equal coverage or simply sacrifice the data in those nodes
		- can be set to `valid` or `same`
			- valid will lose nodes
			- same will pad with zeros 
- Convolutional Layers in Keras
	- see snippet 
- Quiz: Dimensionality
- Pooling Layers (reducing dimensionality)
	- Take covolutional layers as input
	- convolutional layers are also feature maps and require a large number of filters
	- higher dimensionality means more parameters which can lead to overfitting
	- pooling layers help control dimensionality
	- Max pooling layers take a stack of feature maps as input
		- Computes a small window size and stride to reduce dimensionality
		- moderate reduction in size (half as tall and half as wide)
	- Global Average Pooling Layer
		- Each feature map is reduced to a single value
		- From 3d array to a vector
- CNNs for Image Classification
	- Color images have a depth of 3 (rgb), black and white has a depth of 1
	- Convolutional layers will be used to make the array deeper as it moves through the sequence
	- Max pooling layers will be used to decrease the  spacial diumensions
	- Should discover hierarchies of spacial patterns
	- padding is better when set to same, and strides to 1
	- We'll want to increase filters as layers increase
	- relu activation in all layers
	- witout max pooling we will only increase the depth of the array without modifying width and height
	- to reduce the height and width, we add max pooling layers
	- goal is an array that is quite deep but relatively small height and width
	- This will gradually takes spatial data and record the contents of the image, this removes the need for pixels to remain together in order to be interpreted
	- Finally we can flatten and feed to one or more fully connected layers
	- reminder: output layers always match number of classes 
	- Things to Remember
		- Always add a ReLU activation function to the Conv2D layers in your CNN. With the exception of the final layer in the network, Dense layers should also have a ReLU activation function.
		- When constructing a network for classification, the final layer in the network should be a Dense layer with a softmax activation function. The number of nodes in the final layer should equal the total number of classes in the dataset.
		- Have fun! If you start to feel discouraged, we recommend that you check out Andrej Karpathy's tumblr with user-submitted loss functions, corresponding to models that gave their owners some trouble. Recall that the loss is supposed to decrease during training. These plots show very different behavior :).

- CNNs in Keras: Practical Examples
	- CIFAR 10 Database 
- Mini Project: CNNs in Keras
- Image Augmentation in Keras
	- Invariant representation of the image
	- Size, position, or angle should not matter 
	- Scale Invariance (size)
	- Rotation Invariance (angle)
	- Translation Invariance (shifting)
		- Max pooling layers allows for translation invariance
		- Computers can only see a matrix of pixels
		- Data augmentation (invariance and prevents overfitting)
			- Rotation Invariance, add images to training set with random rotation
			- Translation, add random translations
			- use `datagen.flow` to add training subset with augmented images
			- Note on steps_per_epoch

				- Recall that fit_generator took many parameters, including
				
				```
				steps_per_epoch = x_train.shape[0] / batch_size
				```
				
				- where x_train.shape[0] corresponds to the number of unique samples in the training dataset x_train. By setting steps_per_epoch to this value, we ensure that the model sees x_train.shape[0] augmented images in each epoch.

- Mini Project Image Augmentation
- Groundbreaking CNN Architecture
	- ImageNet Project
	- AlexNet Architecture
		- pioneered ReLU and dropout
	- VGG Architecture (16, 19)
		- Simple and elegant
		- 3x3 convolutions
		- 2x2 pooling
		- 3 fully connected layers
	- Resnet Achitecture
		- Similar to VGG
		- Largest has 152 layers
		- Vanishing Gradients problem which arises when we train using backpropagation
		- Solutions was to skip layers 
		- These architectures are available as APIs in keras  
- Visualizing CNNs 1
	- Visualizing activation maps to understand how CNNs are working
	- Start with an image containing random noise and gradually add filter depth until the output becomes increasingly complex
	- Using an image instead of random noise creates an interesting hybrid between the filter and the input
	- https://experiments.withgoogle.com/what-neural-nets-see
- Visualizing CNNs 2
	- Visualization of CNN allow us to see exactly how it activates (or what it sees) 
- Transfer Learning
	- Transfer learning involves taking a pre-trained neural network and adapting the neural network to a new, different data set.
	- One technique is where only one layer in a convolutional network is trained
	- Thrun used a transfer learning approach with an inception model. 
	 - As final step he replaced the fully connected layer with another more simple layer
	 - inputs for each layer were pre-trained on image net
	 - this works best when the dataset large and the data is very close the MNIST database
	 - There are four main cases:

		1. new data set is small, new data is similar to original training data
		2. new data set is small, new data is different from original training data
		3. new data set is large, new data is similar to original training data
		4. new data set is large, new data is different from original training data
	- Overfitting is a concern when using transfer learning with a small data set.
	- See PDF for full detail
- Transfer Learning in Keras
	- Classifying dog breeds 
		- dataset 8000+ images
		- VGG-16  model pre-trained 
		- slice off end of the network and add a new classification layer, freeze weights in other layers and only train layers of the new classification layer.  
		- We'll save output as new input
		- finally we'll have only two layers, input and output
		- new dataset will be known as bottleneck features
		- use globalaveragepooling layer for dimensionality reduction

# Deep learning for cancer detection

- 5.4 million cases per year
- 20% of americans will eventually get skin cancer
- the probability of stage 4 patients surviving (over 5 years) is very low
- 130,000 images of skin conditions (labeled and biopsied). We had ground truth classification of all the types of diseases of the skin.
- Architecture
	- Recurrent Convolutional Neural Network
	- 757 classes
	- Random vs Pre-initialized weights
		- Significantly better results when the net was trained from completely different images
	- validating the training
		- subset for testing, 72% accuracy on a three way classification
		- Dermatologist could not meet this accuracy
	- Precision vs Recall, Sensitivity Specificity
		- Sensitivity
			- Of all the sick people, how many did we diagnose as sick?
			- Out of all the malignant lesions, what percentage are to the right of the threshold (correctly classified)?
		- Specificity
			- Of all the healthy people, how many did we diagnose as healthy?
			- Out of all the benign lesions, what percentage are to the left of the threshold (correctly classified)?
- Threshold value for classifying melanoma should be low. It's better to send a healthy person for furthur diagnosis then to send a sick person home.
- Refresher on ROC
	- Perfect, Good, and Random Split (1, 0.8, 0.5)
	- Calculate True Positives, False Positives
	- We can calculate for every possible split
	- We take those calculations and create a curve
	- The closer the area under the ROC curve is to 1, the better your model is.
	- And we plot that point, where the coordinates are (Sensitivity, Specificity). If we plot all the points corresponding to each of the possible thresholds between 0% and 100%, we'll get the ROC curve that I drew above. Therefore, we can also refer to the ROC curve as the Sensitivity-Specificity Curve.
	- A completely random algorithm will produce a straight line as its ROC curve
- Visualization
	- complex n-dimensional outputs and display them in 2d space
	- T-SNE Visualization
- What is the netwok looking at?
	- Skin patches with darker sections
- Refresher on Confusion Matrix
	- Table to store four values TN, TP, FP, FN
	- Type 1 and Type 2 Errors
		- Sometimes in the literature, you'll see False Positives and True Negatives as Type 1 and Type 2 errors. Here is the correspondence:
		
		- Type 1 Error (Error of the first kind, or False Positive): In the medical example, this is when we misdiagnose a healthy patient as sick.
		- Type 2 Error (Error of the second kind, or False Negative): In the medical example, this is when we misdiagnose a sick patient as healthy.    
- Multi-class Confusion Matrix
	- Dermatologist have a higher confusion value than neural networks  

- Some additional definitions and resources

- In a fully connected layer, each neuron receives input from every element of the previous layer. In a convolutional layer, neurons receive input from only a restricted subarea of the previous layer. ... The input area of a neuron is called its receptive field.

- https://medium.com/@sh.tsang/review-inception-v3-1st-runner-up-image-classification-in-ilsvrc-2015-17915421f77c

- https://towardsdatascience.com/understanding-your-convolution-network-with-visualizations-a4883441533b

- https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e

- https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8

- https://ayearofai.com/rohan-4-the-vanishing-gradient-problem-ec68f76ffb9b 

- https://towardsdatascience.com/real-time-multi-facial-attribute-detection-using-transfer-learning-and-haar-cascades-with-fastai-47ff59e36df0 

- https://jhui.github.io/2017/03/16/CNN-Convolutional-neural-network/

- https://www.dlology.com/blog/one-simple-trick-to-train-keras-model-faster-with-batch-normalization/

- https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#elu

## Image Augmentation in Keras

In order to make the most of our few training examples, we will "augment" them via a number of random transformations, so that our model would never see twice the exact same picture.

https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# Reinforcement Learning

- Intro
	- Study environments with well defined rules and dynamics
	- construct algorithms to understand rules and logic 
	
- Applications
	- Self Driving Cars, Ships, Airplanes
	- Board Games 
		- (TD Gammon)
		- Alpha Go
			- More configurations in this game than atoms 
	- Video Games
		- Atari Breakout
		- Dota
	- Robotics
		- teaching robots to walk   
- Setting
	- Learning from interaction
	- Agent - learner or decision maker
	- beginning from a random choice of action, learning from the response/feedback
	- In general, the goal is initially only to maximize reward 
	- Systematically proposing and testing hypotheses in real-time. 
	- Exploration-Exploitation Dilemma
		- Exploration
			- Exploring potential hypetheses for how to choose actions
		- Exploitation
			- Exploiting limited knowledge about what is already known should work well
	- Maximizing total rewards over an agents lifecycle   
- OpenAI Gym
	- An open source toolkit for developing and comparing RL algorithms
		- frozen lake environment
		- blackjack
		- small world/large cliff
		- taxi world  
- Resources
	- Classical algos in RL
	- Free text book 
- Reference Guide

# The RL Framework

- Intro
	- Goal is to formulate a real world problem to be solved with RL
- The setting 
	- Agent-Environment Relationship
	- Time evolves in a time-step
	- observation, reward, action
	- We'll always assume that the agent can fully observe its environment state
	- ith state informs actions and rewards
	- Agent has the goal to maximize cumulative rewards over all time steps 
- Episodic vs. Continuing Tasks
	- Episodic tasks have a well defined endpoint (return total rewards at time step T)
	- New episodes learns from previous episodes
	- Continuing tasks go on for infinity
	- We used discounted returns to calculate cumulative rewards 
- Sparse rewards
	- reward signal is largely uninformative (e.g, reward at the end of a chess game with no feedback about a particular move or set of moves) 
- The reward hypothesis
	- Reward Hypothesis/Goal - maximizing expected cumulative rewards
	- Rewards can be highly subjective
- Goals and rewards, Part 1
	- Google Deep Mind, teaching a robot to walk
	- detailing the actions
		- forces that the robot applies to its joints 
	- states
		- current positions and velocities of the joints
		- measurement of the ground
		- contact sensor data
	- based on the information in the state, the agent must select a different action 
- Goals and rewards, Part 2
	- Detailing States and Actions
	- Specifying Rewards
		- at every time step the agent received a reward porportional to its forward velocity
		- penalized for torques against each joint
		- rewards for direction and balance on the center of the track
		- it was framed as an episodic task where the episodes terminated when the robot fell 
		- in summary, the goals were for the agent to walk fast, forward, smoothly, and for as long as possible 
		- deepminds hypothesis pointed to the idea that the reward could be very simple
- Cumulative Reward
	- agent cannot focus on individual time steps
	- actions have short and long-term consequence
	- agent needs to gauge the effect of long-term consequence
	- Agent is looking to maximize expected(predicted) return (maximizing the sum)
- Discounted Return
	 - value rewards that come sooner more highly especially when the probability of future rewards is variable
	 - we will define a larger discount rate to apply to returns that will occur sooner and are more likely as opposed to ones that occur later and are less likely
	 - we use gamma notation for discount rate and we apply an exponent to successive gammas applied to create a decay across the returns
	 - gamma should be a value between 0 and 1
	 - discounting is particulary relevant to continuing tasks since features are infinite 
- MDPs, Part 1
	- Markov Decision Process
	- Set of possible actions as the action space (denoted with a scripted A)
	- Set of possible states as the state space (denoted with a script S (nonterminal) and S+ (all states including terminal))
- MDPs, Part 2
	 - Denoting rewards at different states
	 - the environment uses very little information to make its decisions, the envrironment response does not look backwards 
- MPDs, Part 3
	- Formally, an MDP is defined by the set of states, actions, rewards, one-step dynamics, and the discount rate
	- discount factor should be less than 1, commonly set to 0.9 to prevent looking infinitely into the future. This prevents short sightedness
	- For real world problems, you'll formally decide the MDP 
	- Agents knows states and actions and discount
	- Agent does not know one step dynamics
- Finite MDPs
	- both the state space and the action space must be finite 
- Summary
	- see pdf 

# The RL Framework: The Solution

- Policies
	- Reward is always decided in the state of the decision as well as the state that follows
	- Simplest policy is a mapping from state to action
	- deterministic policy : input is state, output is action (represented by pi)
	- stochastic policy : mapping accepts and environment state s and action a and returns probability that agent takes action a while in state s
	- any deterministic policy can be expressed as probability as well (0,1)
- Gridworld Example
	- A gamified example 
	- episodic task 
	- reward signal punishes the agent for every time step away from the goal, rewards spike at completion 
- State-Value Functions
	- A function of the environment state
	- each state has a corresponding number
	- each state yields the return that's likely to follow if the agent starts in that state and then followed the policy for each time step
		- for each state s 
		- it yields the expected return 
		- if the agent starts in state s 
		- and then uses the policy
		- to choose its actions for all time steps  
- Bellman Equations
	- You don't need to look calculate the sum of states every time
	- instead, we'll find that the value function has a nice recursive property
	- We can use the most current value because it corresponds to sum of all of the rewards to the end
	- The value of any state is the sum of the immediate reward + the sum of state that follows (discounted value)
	- Calculate the expected value of the sum 
	- immediate reward and dicounted state to follow cannot be known with certainty
	- we can express the value of any state in the mdp in terms of the immediate reward and the discounted state that follows

> All of the Bellman equations attest to the fact that value functions satisfy recursive relationships


- Optimality
	- Value of policy pi prime is always larger that value of policy pi
	- greater expected return makes for a better or policy
	- a policy pi prime is better than or equal than policy pi if the value of pi prime when state is applied and value is greater for all states
	- optimal state-value function is denoted as v*
- Action-Value Functions
	- lowercase q instead of v
	- a function of the environment state and the agent action
		- for each state s and action a
		- it yields the expected return
		- if the agent starts in state s
		- then chooses action a
		- and the uses the policy
		- to choose its actions for all time steps 
	- we need up to four values for each state corresponding to a different action
	- v* or q* for optimal state
- Optimal Policies
	- The agent interacts with the environment and from that interaction is estimates the optimal value-action function and find corresponding optimnal policy 
	- agent should pick the action that yields the highest exepcted reward value
	- if the agent has the optimal value function it can quickly obtain the optimal policy
- Summary
	- See PDF

# Dynamic Programming

- OpenAI Gym: FrozenLakeEnv
	- Assumption is that agent knows everything about the environment
	- Finding the optimal policy 
- An iterative method, part 1
	- evaluating policies 
		- enumerate the states 
		- applying the bellman equation for each state (except the terminal state which is always 0)
		- because we have the value of the terminal state we can solve the system of equations
		- iterative method
			- start with a guess for the value of each state (typicall zero)
			- improve our guess for the value of each state and adapt the bellman equation as an update rule
			- iterating over each state we increasingly arriving at a better guess for the value of each state 
			- this will yield an estimate that converges to the true value function
			- known as iterative policy evaluation
- iterative policy evaluation
	- assumes agent has perfect knowledge of the environment MDP
	- system of equations motivated by bellman
	- one equation for each environment state
	- relating the value of its state to the value of its successor states
	- again, iterative method will adapt bellman equation as an update rule
	- loops over state, updates values for each state
	- stopping short of true convergence
	- we can stop when updates are hardly noticeable and don't change the estimates of the value function. 
	- Apply stopping criteria 
		- initialize with a small positive number (theta) 
		- if the maximum change over all states is less than the small number theta we set, we can stop
	- we can apply the estimated value of each state to the bellman equation to determine if we've arrived at the perfect value function
- implementation
- action values
	- we can express the value of the state-action pair s_1, right as the sum of two quantities: (1) the immediate reward after moving right and landing on state s_2, and (2) the cumulative reward obtained if the agent begins in state s_2  and follows the policy. 
- policy improvement
	- remember that policy evaluation requires full knowledge of the environment 
	- evaluation and improvement can work together
	- improvement creates a new instance of the policy, which is then re-evaluated and so on until convergence to the optimal policy
	- we always begin with equiprobable value policy
	- break improvement into two steps
		- first calc action value from state value
		- pick one action that maximizes the action value function
	- any policy that (for each state) assigns zero probability to the actions that do not maximize the action-value function estimate (for that state) is an improved policy. 
	- When there are more than one maximized action values, you can choose arbitrarily or use a stochastic approach
- Policy Iteration
	- Combines evaluation and improvement
	- repeat this loop until finally we encounter an improvement step that doesn't require any change to policy
	- guaranteed convergence to the optimal policy in a finite mdp
- Truncated Policy Iteration 
	- using an absolute number of iterations instead of using of waiting for natural termination
	- we don't need a perfect or near-perfect idea of the value function to get an optimal policy 
- Value Iteration
	- Policy evaluation step is stopped after a single sweep
	- Combining the two equations to simplify iteration
- Summary
	- Policy Iteration
		- Finds the optimal policy through successive rounds of evaluation and improvement.
	- Policy Improvement
		- Given a value function corresponding to a policy, proposes a better (or equal) policy.
	- (Iterative) Policy Evaluation
		- Computes the value function corresponding to an arbitrary policy.
	- Value Iteration
		- Finds the optimal policy through successive rounds of evaluation and improvement (where the evaluation step is stopped after a single sweep through the state space).     

# Monte Carlo Methods

- Not given environment knowledge and must learn from interaction

- MC Prediction: State Values
	- Given a policy, how might the agent estimate the value function
	- On-Policy Method
		- Generate episodes from following policy pi. Then use the episodes to estimate v pi
	- We'll use some sample episodes to estimate the value function
	- Look at all the occurences with state X
	- Calculate discounted returns after occurences of state x
	- MC Prediction takes the average of the values and plugs it in as estimate for the value of state x 
	- The value of the state is the value of the expected return after that state has occured
	- Visit: Every occurence of state in an episode as a visit
	- First visit: The value of state Y is an average of the returns after first visit
	- Every vist: an average of the returns after every visit

- MC Prediction: Action Values
	- In DP, we use state value function to obtain action values, but here we don't have one-step dynamics
	- To get action values, we make a small modification to DP algo
	- We'll look at the visits to each state-action pair
	- We'll again use first vist or every visit methods
	- Our algo can only estimate from pairs that have been visited
	- We don't try to evaluate for determistic polices and only for stochastic policies
	- We can calculate a nice action value as long as the agent visits enough states

- Generalized Policy Iteration
	- the commonalities of policy iteration, truncated policy iteration, value iteration (see screengrab for refresher)

- MC Control: Incremental
	- Again, evaluation followed by improvement
	- Instead of calculating averages at the end of episodes, we update at each visit
	- This algorithm can keep a running estimate of the mean of a sequence of numbers
	
- MC Control: Policy Evaluation
	- Algo looks at each visit in order and updates each successively updates the mean mu for a single state action pair, but we want to maintain values for many state-action pairs
	- Agent samples an episode
	- For every time step, we look at corresponding state-action pair 
	- first visit, we calc corresponding return
	- from there we update the corresponding value
	- We have to initialize with the number of times we visited each pair

- MC Control: Policy Improvement
	- We have to ammend the DP algo (which used greedy policy)
	- Instead of always constructing a greedy policu, we'll construct a stochastic policy that selects a policy closest to the greedy policy
	- epsilon greedy policy should be included in the improvement step

- Exploration vs Exploitation
	- A successful RL agent cannot act greedily at every time step (that is, it cannot always exploit its knowledge); instead, in order to discover the optimal policy, it has to continue to refine the estimated return for all state-action pairs (in other words, it has to continue to explore the range of possibilities by visiting every state-action pair). That said, the agent should always act somewhat greedily, towards its goal of maximizing return as quickly as possible. This motivated the idea of an \epsilonϵ-greedy policy.  
	- setting \epsilon = 1ϵ=1 yields an \epsilonϵ-greedy policy that is equivalent to the equiprobable random policy.
	- At later time steps, it makes sense to favor exploitation over exploration, where the policy gradually becomes more greedy with respect to the action-value function estimate. 
	- Greedy in the Limit with Infinite Exploration (GLIE)
		- modify the value of \epsilonϵ when specifying an \epsilonϵ-greedy policy. In particular, let \epsilon_iϵ 
i
​	  correspond to the ii-th time step.
	- Setting the Value of \epsilonϵ, in Practice
		- Since we can't have infinite episodes in practice, we can use fixed epsilon or let epsilon decay to a small positive number like 0.1
		- This is because one has to be very careful with setting the decay rate for \epsilonϵ; letting it get too small too fast can be disastrous. If you get late in training and \epsilonϵ is really small, you pretty much want the agent to have already converged to the optimal policy, as it will take way too long otherwise for it to test out new actions!
​
	- MC Control Constant-alpha
		- Calculating error by looking at the estimate versus the actual
		- If error is > 0, we increase the function
		- if error is < 0, we decrease
		- we change by an amount proportional to learning rate (alpha)
		- This ammendment should be made to MC policy evaluation
		- This ensures that the agent primarily considers the most recently sampled returns when estimating the action-values and gradually forgets about returns in the distant past. "taking a forgetful mean of a sequence"
		- You should always set the value for \alphaα to a number greater than zero and less than (or equal to) one.

			1. If \alpha=0α=0, then the action-value function estimate is never updated by the agent.
			
			1. If \alpha = 1α=1, then the final value estimate for each state-action pair is always equal to the last return that was experienced by the agent (after visiting the pair).

			1. Smaller values for \alphaα encourage the agent to consider a longer history of returns when calculating the action-value function estimate. Increasing the value of \alphaα ensures that the agent focuses more on the most recently sampled returns. 
		
		- It is important to mention that when implementing constant-\alphaα MC control, you must be careful to not set the value of \alphaα too close to 1. This is because very large values can keep the algorithm from converging to the optimal policy \pi_*π 
∗
​	 . However, you must also be careful to not set the value of \alphaα too low, as this can result in an agent who learns too slowly. The best value of \alphaα for your implementation will greatly depend on your environment and is best gauged through trial-and-error.

# Temporal-Difference Methods

- Introduction
	- Agents will learn online streaming data in real-time, not episodic tasks. 
	- While MC needed an episode to end in order to calculate returns
	- At every move, estimating the probability that it's winning the game as opposed to waiting until the end of the game
	- can solve continuous and episodic tasks

- TD Prediction
	- This algorithm is a solution as long as we never change policy between episodes
	- Adapting the update step 
	- Using the Belman equation to motivate the adjustment to the update step
	- Understand the value of state in terms of the values of its successor states
		- removing any mention of the return that comes at the end of the episode
		- values are being updated along the way
	- TD target
		- finds middle ground between the previous estimate and the next
	- One-Step TD or TD(0)
		- updating the value function after individual step for the previous state
		- (continous tasks) as long as the agent interacts with the environment for long enough, we should have a decent estimate for the value function
		- (episodic tasks) check at every time step if the current state is the termal state
	- Comparing to Monte Carlos
		- Whereas MC prediction must wait until the end of an episode to update the value function estimate, TD prediction methods update the value function after every time step. Similarly, TD prediction methods work for continuous and episodic tasks, while MC prediction can only be applied to episodic tasks.
		- In practice, TD prediction converges faster than MC prediction.
	- Summary
		- Agent interacts at the environment
		- recieves some state at s0
		- chooses an action based on a policy
		- immediately receives reward and next state
		- it uses that information to update the value function for the previous state   
- TD Prediction: Action Value
	- Uses state, action pairs to estimate the action value function 
	- Updating the action value function after each action is chosen 
- TD Control: Sarsa(0)
	- Select action at every time step using a value that's epsilon greedy
	- Sarsa for short, each action value update uses a state action reward, next state, next action tuple of interaction
	- Summary
		- Sarsa(0) is guaranteed to converge to the optimal action-value function, as long as the step-size parameter \alphaα is sufficiently small, and the Greedy in the Limit with Infinite Exploration (GLIE) conditions are met. The GLIE conditions were introduced in the previous lesson, when we learned about MC control. Although there are many ways to satisfy the GLIE conditions, one method involves gradually decaying the value of \epsilonϵ when constructing \epsilonϵ-greedy policies.
		- then the algorithm is guaranteed to yield a good estimate for q_ as long as we run the algorithm for long enough. 
- TD Control: Sarsamax (Q-learning)
	- After receiving the reward we're update the policy before choosing the next action
	- for every step use an action from the greedy policy instead of an epsilon greedy policy
	- directly attempts to estimate the optimal action value function at every time step
- TD Control: Expected Sarsa
	- Only difference from sarsa max is in the update step for the action value
	- Sarsa-max took the maximum over all actions of all possible next state action pairs
	- Expected sarsa uses the expected value of the next action pair, where the expectation takes into account the probability that the agents selects each possible action from the next state.

- Analyzing Performance
	- The differences between these algorithms are summarized below:

		- Sarsa and Expected Sarsa are both on-policy TD control algorithms. In this case, the same (\epsilonϵ-greedy) policy that is evaluated and improved is also used to select actions.
		- Sarsamax is an off-policy method, where the (greedy) policy that is evaluated and improved is different from the (\epsilonϵ-greedy) policy that is used to select actions.
		- On-policy TD control methods (like Expected Sarsa and Sarsa) have better online performance than off-policy TD control methods (like Sarsamax).
		- Expected Sarsa generally achieves better performance than Sarsa.
	- If you would like to learn more, you are encouraged to read Chapter 6 of the textbook (especially sections 6.4-6.6).

# RL in Continuous Spaces
 
1. Deep Reinforment Learning
	- Refers to approaches that use deep learning, mainly, Multi-Layer Neural Networks to solve reinforcement learning problems.
	- RL in Continous Spaces
	- Deep Q-Learning
	- Policy Gradients
	- Actor-Critic Methods
2. Resources
	- Sutton/Barto Part II
3. Discrete vs. Continous Spaces
	- Discrete Spaces
		- Finite set of states and actions (chess, or grid-based)
		- allows us to use a dictionary to map every state action pair to a real number
		- Critical to algos like value iteration that iterate over each possible state or action
	- Continuous Spaces
		- not restricted to a set of distinct values
		- can accept a range of values (real numbers)
		- can be multi-dimensional
	- Why Continuous?
		- Because the real-world has physics
		- most physical environments require continous learning
4. Space Representation
	- Check out this [table of the environments](https://github.com/openai/gym/wiki/Table-of-environments) available in OpenAI Gym. Here Discrete(...) refers to a discrete space, and Box(...) indicates a continuous space. See [documentation](https://gym.openai.com/docs/#spaces) for more details. 
5. Discretization
	- Converting a continuous space into a discrete one
	- identifying certain position/states as relevant (even in a continuous environment)
	- actions can be discretized as well
	- occupancy grid can be used to identify obstacles
	- binary space partitioning or quad trees
- Non-Uniform Discretization
	- Speed for example, can be discretized into ranges of different lengths (non-uniform) 
6. Tile Coding
	- Only with prior knowledge of the state space can you manually design a discretisation scheme
	- In order to function in arbitrary environments we need a more generic method.
	- Overlay multiple grids or tiling over the state space (slightly shifted)
	- Modifying value function with weight
	- tile params must be manually selected ahead of time (splits, sizes, etc)
	- Adaptive tile coding will split the state space more automatically (splitting when the value function isn't changing), uses heuristics not relying on a human to define discretization ahead of time
7. Coarse Coding
	- sparser set of features to encode the state space
	- binary bit vector
	- spherical grid
	- smaller circles results in less generalization
	- large circles lead to more generalization and a smoother action value function
	- Radial Basis Functions
	- result is a binary vector
	- the distance from the center of the circle is a measure of how active that circle is
	- RBFs can reduce drastically the number of features
8. Function Approximation
	- when discretizing large continous spaces, the number of discrete spaces will become very large
	- introduces a parameter vector w, tuning until we find the closest approximatoin for action/state value function
	- We'll need a feature vector (which allows us to used derived values)
	- we use dot product to create a scalar
	- we're trying approximate the underlying value function with a linear function
9. Linear Function Approximation
	- For example, we can use gradient descent to approximate the underlying value function
	- minimizing the difference between the true value function and the approximation 
	- minimize error, error gradient or derivative, update rule
	- after sampling enough states we can come close to the actual value
	- changing weights a small step away from the vector
	- for model-free we need to approximate the action value function (same gradient descent method)
	- to compute all the action values at once, we can compute an action vector (taking in both state and action)
	- extend our weight vector into a matrix
	- this is called Action Vector Approximation
	- limitations
		- can only represent linear relationships between inputs and outpus
		- for non-linear shapes, [obviously] we need non-linear functions 
10. Kernel Functions
	- Again, we need feature transformation taking in state and action pairs to create a feature vector
	- Kernel Functions or Basis functions
		- transform input state into a different space
		- this allows to use linear functions approximation
		- radial basis function
			 - for any given state we can reduce the state  representation to a vector of responses from these radial basis functions, then we can continue using linear function approximation
11. Non-Linear Function Approach
	- we can capture non-linear relationships using abitrary kernels
	- but what if our underlying values were truly non-linear
	- activation functions can be used, we can update parameters using gradient descent

# Deep Q-Learning

1. Intro to Deep Q-Learning
	- an elegant algorithm that demonstrates how you can use a neural network to estimate value
	- Adapting model-free methods 
2. Neural Nets as Value Functions
	- NNs are universal function approximators
	- input should be a vector (feature transformation)
	- weights of the nn filling the parameter w
	- Steps
		- use the squared difference between the estimated and target value as our error or loss
		- then we back propagate it through the network, adjusting weights along the way to minimize loss (typically applying gradient descent)
		- we need to know the derivative of the value function respective to its weights
		- we just need a way to figure out loss
	- we need a realistic target in place of the true value functions
3. Monte Carlo Learning
	- cumulative discounted reward can be the target
	- in our update rule, substitue the unkown true value function with a return
	- evaluation, generate an episode, update for each step, followed by an improvement step (every-visit)
4. Temporal Difference Learning
	 - uses the td target (or estimated return)
	 - we can use the td target in place of the unknown value function
	 - we can them apply function approximation
	 - use our gradient descent update rule and apply weights 
	 - good for episodic tasks where each episode is guaranteed to terminate
	 - SARSA is an on-policy, mutating the policy along the way, the policy being learned and the one being followed are tightly coupled
5. Q-Learning
	- off policy variant of TD learning
	- main diff is the update step
	- instead of picking the next action, we choose an action greedily (not taking this action but used for performing the update)
	- one policy to take actions, a yet another to perform value updates (e-greedy and greedy)
	- For continuing tasks
		- modify to remove the concepts of episodes
	- Sarsa vs Q (see graphic)
	- Off-policy advantages
		- decoupling the action from the update
		- different variations of the algorithm
		- more exploration when learning
		- learning from demonstration
		- supports offline or batch learning  	 - One drawback of both SARSA & Q-Learning, since they are TD approaches, is that they may not converge on the global optimum when using non-linear function approximation.

6. Deep Q Network
	- DNN acting as a function approximator
	- max value indicating the action to take
	- fed back the state at each time step
	- square images were used to optimize nn on gpu
	- deep q is designed to produce a q value for every forward action in an individual task
	- you can use the vector output stochastically or by using the max
	- convolutional layers used to extract temporal difference
	- network weights can diverge (oscillate), to overcome reseachers came up with techniques 
7. Experience Replay
	- storing the experience tuples at each time step using a replay buffer
	- sample a small batch of tuples, reuse tuples, cut down on costly calculations
	- you can sample at random (out of order)
	- prevents oscillation or divergence
	- RBF kernels or Q-network as function approximators
	- blowing up the order of training, so that learning is more robust, ignores the sequence of observation of tuples
	- reduces reinforcement learning to supervised learning
	- prioritized experience tuples that are rare or more important
8. Fixed Q Targets
	- another kind of correlation that q-learning is susceptible to
	- too much correlation with target and parameters we are changing
	- decoupling the target from the parameters that are changing
	- fixing the function parameters that we don't change during the learning step
	- less likely to diverge or fall into oscillations
9. Deep Q-Learning Algorithm
	- Two main processes
	- **sample** the environment to store into replay memory
	- **learn** from batch of randomly selected samples
	- not dependent on each other
	- circular Q that retains the n most recent experience tuples (memory is finite)
	- for temporal relations stack a few input frames (padding is sometimes required)
	- learning step must wait for sufficient number of samples
	- reward clipping, error clipping, etc.
	- Mnih et al., 2015. [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). (DQN paper)
He et al., 2015. [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852). (weight initialization)

10. DQN Improvements
	- Double DQNs
		- rewrite the target to use argmax
		- accuracy of q-values depends on prior experience 
		- select best action by evaluating first using w and w prime
		- prevents chance selections
		- w- can be reused
	- Prioritize Replay
		- importance experience occur more infrequently
		- older important experience may get lost as buffer is flushed
		- we can use td error detla to assign priority
			- the bigger the error the more we expect to learn from that tuple
			- compute a sampling probability
			- when a tuple is picked, we can update its priority 
			- it td error is 0, priority is also 0, so we can add a small constant to prevent ignoring samples
			- have to make one adjustment to our update rule, sample must match underlying distribution, non-uniform sampling will not match original distribution, so introduce a sampling weight (see prioritized experience paper)
	- Dueling Networks
		-  Using two streams, on to estimate state value, and one that estimates advantage for each action (advantage values)
		-  value of most states don't vary a lot across actions
		-  see Dueling networks paper
11. Implementing Deep Q-Learning
	- [Keras](https://keon.io/deep-q-learning/)
	- [PyTorch](http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
12. TensorFlow Implementation
13. Wrap Up

# Policy-Based Methods

1. Why Policy-Based Methods
	 - Value-Based - policy is defined implicitly
	 - Policy-Based - optimal policy without looking at action value
	- Simplicity
		- directly estimate the optimal policy
		- deterministic would be a simple mapping
		- stochastic would be a probability given state and action
	- Stochastic Policies
	- good non-determinist Environments 
	- Aliased States - two or more states perceived to be identical but are actually different
	- two states with identical features would have equal action, state value, meaning he may become stuck repeating the same actions
	- Well suited for continous action spaces
	- and high dimensional action spaces
2. Policy Function Approximation
	- we can apply function approximation in calculating policy
	- we can use objective functions to find maximum objective value
		- Start State Value
		- Average State Value
		- Average Action Value
		- Average Reward 
3. Stochastic Policy Search
	- changing objective parameters slightly to find better objective value
	- can use any policy function
	- Steepest Ascent approach
	- simulated annealing, reduce noise or radius as we approach an optimal solution
	- Adaptive noise - reduce search radius when we are closer to the optimal policy, or increase search radius from the current best policy
	- may get stuck in local optimum or take long to converge
4. Policy Gradients
	- change policy parameters by a small fraction alpha, of the gradient of the objective function, J theta
	- Compute the gradient of the objective function with respect to policy parameters
	- directly compute the next set of policy parameters that seem most promising
	- iterate while updating gradient
	- gradient can be estimated using finite differences
	- computing gradients analytically
	- Likelyhood Ratio Trick
	- compute the derivative of log probabilities
	- tensorflow, pytorch have this implemented
	- we can update policy parameters to improve policy iteratively
5. Monte Carlos Policy Gradients
	- reinforced algorithm perform update at the end of each episode
6. Constrained Policy Gradients
	- intermediate policies where we've constrain a policy to prevent parameters from being changed too drastically
	- we can also achieve using a penalty
	- paramater difference 
	- policities can be probability distributions
	- KL-Divergence as Constraint
	- Proximal Policy Optimzation
7. Recap
	- Advantages of value-based methods
		- directly map from state to actions
		- good with continous control tasks
		- true stochastic policies 
		- active area of research

# Actor-Critic Methods

1. Keep track of state of state, action values to calculate the objective
2. A better score function
	- something that can be computed online as we interact and does not depend on the end of the episode
	- we can use temporal difference mechanism to update value
3. Two Function Approximators
	- Q-learning with function approximation to update action values
	- we can use two function approximators for policy update and value update
	- Policy (actor), Value (critic)
	- can be trained independently using neural nets
4. The actor and the critic
	- the actor performs, while the critic provides feedback
	- iteration can happen until not much improvement is seen
	- critic can give better and better feedback as time goes on
	- At each time step we sample the current state and estimate an action which is take by the actor, the critic evalutes, the actor then updates it value, then critic updates its value at the end of each time step
5. Advantage Function
	- reduce variance between update steps
	- advantage value tells us how much we gain from taking some action
6. Actor-Critic with Advantage
	- replace state action value with advantage value
	- critic needs to keep track of two value functions, but can use td error to calculate advantage
7. Summary
	- Variant of policy and value based methods
	- RL used to cool Googles data center by 40 percent
