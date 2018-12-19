import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Reading the csv file
data = pd.read_csv("data.csv")

# Splitting the data into X and y
import numpy as np
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])



svc = SVC()
dt = DecisionTreeClassifier()

# Use the train_test_split function to split the data into
# training and testing sets.
# The size of the testing set should be 20% of the total size of the data.
# Your output should contain 4 objects.
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)

dt.fit(X_test,y_test)
svc.fit(X_train,y_train)
