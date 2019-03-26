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