# Decision tree classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree, datasets
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import ShuffleSplit, GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt

iris = datasets.load_iris()
dt = DecisionTreeClassifier(random_state=1)
x_train, x_valid, y_train, y_valid = train_test_split(iris.data,iris.target,random_state=1,test_size=0.20)
p_grid = {'max_depth':[2,3,4,5,10,15,20,25,30]}
inner_cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=1)
clf=GridSearchCV(estimator=dt,param_grid=p_grid,cv=inner_cv,verbose=0)
clf.fit(x_train,y_train)
print(clf.best_estimator_)

# Model testing
predictions = clf.predict(x_valid)
# Performance evaluation
acc = accuracy_score(y_valid, predictions)
print(acc)
print(classification_report(y_valid,predictions))
cm = confusion_matrix(y_valid, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
#plt.show()

# Model interpretability
dtopt = DecisionTreeClassifier(random_state=1,max_depth=clf.best_estimator_.max_depth)
clf2 = dtopt.fit(x_train,y_train)
tree.plot_tree(clf2)
print(clf2.feature_importances_)

