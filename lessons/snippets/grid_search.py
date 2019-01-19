from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.svm import SVC

parameters = {'kernel':['poly', 'rbf'],'C':[0.1, 1, 10]}

# Create a scorer
scorer = make_scorer(f1_score)

clf = SVC()

# Create the object.
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# Fit the data
grid_fit = grid_obj.fit(X, y)

# Get the best estimator.
best_model = grid_fit.best_estimator_
best_model.fit(X, y)
best_model.predict(X)
