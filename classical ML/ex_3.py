#Random Forest is an ensanble methods which builds multiple tree and merge them to have a more accurate prediction
#The hyperparameters are the number of estimators (that is the numbero of tree of the forest) and the max_depth (i.e. how much every tree need to be grown: number of decision nodes)
#we can do a manual parameter tuning or a grid search

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics, datasets
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import cv2

iris = datasets.load_iris()
df_feature = iris.data
#print(df_feature)
df_label = iris.target
#print(df_label)

#Manual Hyperparameters tuning

    #splitting training/testing
    #freezing the test set
    #splitting training/validation (to retrive a validation set for hyperparameters tuning)
    #defining hyperparameters to be tuned (in this case 2 and 2 hyperparameters)
    #initializing 4 classifiers with the 4 combinations of the hyperparameters
    #training 4 classifiers
    #testing the 4 classifiers performance on the validation set
    #choosing the best combination of hyperparameters

x_train, x_valid, y_train, y_valid = train_test_split(df_feature,df_label,test_size=0.20)

#Training the different classifiers (i.e., the same classifier with different combination of hyperparameters)ù
    #a classifier is a type of machine learning algorithm used to assign a class label to a data input
clf1 = RandomForestClassifier(n_estimators=100,max_depth=2)
clf2 = RandomForestClassifier(n_estimators=100,max_depth=3)
clf3= RandomForestClassifier(n_estimators=200,max_depth=2)
clf4 = RandomForestClassifier(n_estimators=200,max_depth=3)

tclf1 = clf1.fit(x_train,y_train)
tclf2 = clf2.fit(x_train,y_train)
tclf3 = clf3.fit(x_train,y_train)
tclf4 = clf4.fit(x_train,y_train)

pred1 = tclf1.predict(x_valid)
pred2 = tclf2.predict(x_valid)
pred3 = tclf3.predict(x_valid)
pred4 = tclf4.predict(x_valid)

print(accuracy_score(y_valid,pred1))
print(accuracy_score(y_valid,pred2))
print(accuracy_score(y_valid,pred3))
print(accuracy_score(y_valid,pred4))

#We finally test the performance of the best classifier after hyperparameters tuning on the test set
cm1 = confusion_matrix(y_valid, pred1)
cm2 = confusion_matrix(y_valid, pred2)
cm3 = confusion_matrix(y_valid, pred3)
cm4 = confusion_matrix(y_valid, pred4)
cmd1 = ConfusionMatrixDisplay(cm1)
cmd2 = ConfusionMatrixDisplay(cm2)
cmd3 = ConfusionMatrixDisplay(cm3)
cmd4 = ConfusionMatrixDisplay(cm4)
cmd1.plot()
cmd2.plot()
cmd3.plot()
cmd4.plot()
#plt.show()

# Hyperparameters tuning: a more complex but accepted scenario
# Learning the parameters of a function and testing it on the same data is a methodological mistake: a model that would just repeat
# the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data.
# This situation is called overfitting
# To avoid it, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data set as a testset.
# When evaluating different settings ("hyperparameters") for estimators, there is still the risk of overfitting on the test set bacuse the 
# parameters can be tweaked until the estimators performs optimally.
# This way, knowledge about the test set can "leak" into the model and evaluation metrics no longer report on generalization performancce.
# To solve this problem, yet another part of the dataset can be held out as so-called "validation set": training proceeds on the training set
# after which evaluation is done on the validation set, and the experiment seems to be successful, final evaluation can be done on the test set.
# However, by partitioning the available data into three, we drastically reduce the number of samples which can be used for learning the model,
# and the results can depend on a particular random choice for the pair (train, validation) sets.
# A solution to this problem in CROSS VALIDATION. A test set should still be held out for final evaluation, but the validation set is no longer needed
# when doing cross validation. In the basic approach, called k-fold CV, the training set il split into k smaller sets.
# The following procedure is followed for each of the "k-folds":
#       A model is trained using k-1 of the folds as training data;
#       The resulting model is validated on the remaining part of the data (i.e. it is used as a test set to compute a performance measure such as accuracy)
# 
# The performance measure reported by k-fold CV is the average of the values computed in the loop

loop = KFold(n_splits=3, shuffle=True, random_state=1)
clf = RandomForestClassifier()
# Define search space
grid = dict()
grid["n_estimators"] = [10,100,500]
grid["max_depth"] = [2,4,6]
# Define search for hyperparameters tuning
search = GridSearchCV(clf, grid, scoring='accuracy', cv = loop)
result = search.fit(x_train,y_train)
best_model = result.best_estimator_
#print(best_model)
predictions = best_model.predict(x_valid)
print("prediction of the best model (hyperparametes tuned):",accuracy_score(y_valid, predictions))

#search.cv_results_
#print(search.cv_results_['params'])
#print(search.cv_results_['mean_test_score'])