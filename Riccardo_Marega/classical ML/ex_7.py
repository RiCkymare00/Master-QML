# Repeated k-fold CV
import numpy as np
from sklearn.model_selection import RepeatedKFold, train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn import datasets,svm ,metrics, preprocessing
from sklearn.decomposition import PCA

# Hold out cross validation without hyperparameters tuning
# Hold-out is when you split your dataset into a 'train' and 'test' set. The training set is what the model is trained on, and the test set is used
# to see how well that model performs on unseen data

'''X,y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1,test_size=0.40)
clf = svm.SVC(kernel='linear', C=1).fit(X_train,y_train)
clf.score(X_test,y_test)
y_test'''

# 5-fold cross validation without hyperparameter tuning
# Cross-validation or 'k-fold cross-validation' is when the dataset is randomly split up into 'k' groups, One of the groups is used as the test set
# and the rest are used as the training set. The model is trained on the training set and scored on the test set. Then the process i repeated
# untilll each unique group has been used as the test set
 
'''clf=svm.SVC(kernel='linear',C=1,random_state=42)
scores= cross_val_score(clf,X,y, cv=5)
print(scores)
print(scores.mean(),scores.std())'''

# If i want to change the scoring metric

'''scores = cross_val_score(clf, X,y,cv=5,scoring='f1_macro')
scores'''

# ...

#Stratified k-fold
X,y = np.ones((50,1)), np.hstack(([0]*45,[1]*5))
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X,y):
    np.bincount(y[train],np.bincount([y[test]]))

print('---')

kf = KFold(n_splits=3)
for train, test in kf.split(X,y):
    print('train - {} | test - {}'.format(
        np.bincount(y[train]),np.bincount(y[test])
    ))



