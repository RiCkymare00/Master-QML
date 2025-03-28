#simple support vector machine (SVM): algorith that search for the optimal hyperplane that separates items from different classes.
#The goal of this algorithm is to optimize two hyperparameters: the kearnel and a regularization parameter.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from sklearn import datasets,metrics
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer #binarize labels in a one-vs-all fashion
from itertools import cycle

iris = datasets.load_iris()
X = iris.data
Y = iris.target
#whithout doing cross validation, we divide our dataset in training set and test set
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2) #test_size, if given in float (should be between 0.0 and 1.0), represents the proportion of the dataset to include in the test split (in this case 20% of the dataset is given for testing)
#print(X_train, X_test, y_train, y_test)
svm = SVC(kernel="linear", C=1, probability=True) #(SVC = Support Vector Classification (algorithm of supervised machine learning))
svm.fit(X_train,y_train)

#Predict from the test dataset
predictedProbability = svm.predict_proba(X_test)
#print(predictedProbability)
predictions = svm.predict(X_test)
print(predictions)
print(accuracy_score(y_test,predictions))

cm = confusion_matrix(y_test,predictions)
cmd = ConfusionMatrixDisplay(cm)
cmd.plot()
#plt.show()

label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)
y_onehot_test.shape

print(roc_auc_score(
            y_onehot_test,
            predictedProbability,
            multi_class = "ovr",
            average = "micro",
        )
    )

fpr, tpr, roc_auc = dict(), dict(), dict()

n_classes = 3
for i in range(n_classes):
    fpr[i],tpr[i],_=roc_curve(y_onehot_test[:,i],predictedProbability[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])
fpr_grid=np.linspace(0,1,1000)
mean_tpr=np.zeros_like(fpr_grid)
for i in range(n_classes):
    mean_tpr += np.interp(fpr_grid,fpr[i],tpr[i]) #linear interpolation

#average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = fpr_grid
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"],tpr["macro"])
print(roc_auc["macro"])

target_names = iris.target_names
fig, ax = plt.subplots(figsize=(6,6))
colors = cycle(["aqua","darkorange","cornflowerblue"])
for class_id, color in zip(range(n_classes),colors):
    RocCurveDisplay.from_predictions(
        y_onehot_test[:,class_id],
        predictedProbability[:,class_id],
        name=f"ROC curve for {target_names[class_id]}",
        color=color,
        ax=ax,
    )
plt.plot([0,1],[0,1], "k--", label= "ROC curve for chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Extention of Receiver Operating Characteristic/nto One-vs-Rest multiclass")
plt.legend()
plt.show()

