import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

cols = ['LSTAT', 'PTRATIO', 'RM', 'MEDV'] # housing price dataset
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)

plt.show()
