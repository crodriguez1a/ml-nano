from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2), n_estimators = 4)
model.fit(x_train, y_train)
model.predict(x_test)
