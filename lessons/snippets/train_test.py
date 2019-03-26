# Import statements
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

# Import the train test split
# http://scikit-learn.org/0.16/modules/generated/sklearn.cross_validation.train_test_split.html


# Read in the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y.
X = data[:,0:2]
y = data[:,2] # TODO visualize this

# From dataframes
#X = data.iloc[:,0:2]
#y = data.iloc[:,2]

# Use train test split to split your data
# Use a test size of 25% and a random state of 42
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)

# Instantiate your decision tree model
model = DecisionTreeClassifier(random_state=33)

# Fit the model with the training data.
model.fit(X_train,y_train)

# Predict the output(y) using the input(x) from the test set
y_pred = model.predict(X_test)

# Calculate the accuracy comparing test output to predicted output.
acc = accuracy_score(y_test, y_pred)
