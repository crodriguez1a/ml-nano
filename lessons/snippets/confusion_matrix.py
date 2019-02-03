from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

# Compute confusion matrix for a model
model = clf_C
cm = confusion_matrix(y_test.values, model.predict(X_test))

# view with a heatmap
sns.heatmap(cm,
            annot=True,
            cmap='Blues',
            xticklabels=['no', 'yes'],
            yticklab
            els=['no', 'yes'])

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title(f'Confusion matrix for:\n{model.__class__.__name__}')
