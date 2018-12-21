from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer, f1_score

parameters = {'kernel':['poly', 'rbf'],'C':[0.1, 1, 10]}

# Create a scorer
scorer = make_scorer(f1_score)

# Create the object.
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# Fit the data
grid_fit = grid_obj.fit(X, y)

# Get the best estimator.
best_clf = grid_fit.best_estimator_
