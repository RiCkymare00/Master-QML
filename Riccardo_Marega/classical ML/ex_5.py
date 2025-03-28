# Logistic regression: is both a statistical and a ML learining algorithm which is used in binary classification problems (e.g. having a disease or not)

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import ShuffleSplit, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

dataset = load_iris()
lr = LogisticRegression()
x_train, x_valid, y_train, y_valid = train_test_split(dataset.data,dataset.target,random_state=1,test_size=0.20)
p_grid_lr = {"C":[5,10,15,20,25,30]}
inner_cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=1)
clf = GridSearchCV(estimator=lr,param_grid=p_grid_lr,cv=inner_cv,verbose=0)
clf.fit(x_train,y_train)
predictions = clf.predict(x_valid)
acc = accuracy_score(y_valid,predictions)
print(acc)
print(classification_report(y_valid,predictions,target_names=dataset.target_names))
cm = confusion_matrix(y_valid,predictions,labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.show()