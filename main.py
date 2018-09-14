from sklearn import svm
from sklearn import datasets
from sklearn import neighbors,linear_model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import ensemble
# import data_preprocessing
import pickle
"""load data and preprocess
"""

# with open('data_wob_100_ex.pkl', 'rb') as f:
dim=200
with open('w'+str(dim)+'.pkl', 'rb') as f:
# with open('data_wob_100_gray.pkl', 'rb') as f:

    X=pickle.load(f)
    yyy=pickle.load(f)



temp=np.zeros((X.shape[0],dim))
for i ,v in enumerate(X):
    #print(len(np.array(v)))
    temp[i]=np.array(v)

X=np.array(temp)
temp_y=np.array(yyy)
y=[]

for yy in temp_y:
    if yy==1:
        y.append(0)
    elif yy==3:
        y.append(4)
    else:
        y.append(yy)
y=np.array(y)
X=X.reshape(X.shape[0],-1)
X = StandardScaler().fit_transform(X)
# X_train, X_test, y_train, y_test = \
    # train_test_split(X, y, test_size=.3,random_state=1)
"""basic ml methods
"""
C=1.0
svc=svm.SVC(kernel='linear', C=C,)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.5, C=C,)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C,)
# svc=svm.SVC(kernel='linear', C=C,probability=True)
# rbf_svc = svm.SVC(kernel='rbf', gamma=0.5, C=C,probability=True)
# poly_svc = svm.SVC(kernel='poly', degree=3, C=C,probability=True)
# lin_svc = svm.LinearSVC(C=C)
knn=neighbors.KNeighborsClassifier()
lr = linear_model.LogisticRegression()
GaussianProcess=GaussianProcessClassifier(1.0 * RBF(1.1))
# GaussianProcess1=GaussianProcessClassifier(1.0 * RBF(1.0))

DecisionTree=DecisionTreeClassifier(max_depth=5)
RandomForest=RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
MLP=MLPClassifier()
MLP2=MLPClassifier(alpha=0.1)
MLP3=MLPClassifier(hidden_layer_sizes=(50,100,200),alpha=0.1,)

AdaBoost=AdaBoostClassifier()
Gaussian=GaussianNB()
QuadraticDiscriminant=QuadraticDiscriminantAnalysis()
fun_list=[
    ("svc",svc),
    # ("rbf_svc",rbf_svc),
    # ("poly_svc",poly_svc),
    # ("lin_svc",lin_svc),
    # ("knn",knn),
    ("lr",lr),
    ("GaussianProcess",GaussianProcess),
    # ("GaussianProcess1",GaussianProcess1),

    # ("DecisionTree",DecisionTree),
    # ("RandomForest",RandomForest),
    # ("MLP",MLP),
    ("MLP2",MLP2),
    # ("MLP3",MLP3),

    # ("AdaBoost",AdaBoost),
    # ("Naive Bayes",Gaussian),
    # ("QDA",QuadraticDiscriminant)

]
"""ensemle methods
"""
m=ensemble.Ensemble(fun_list)


m.crossValidation(X,y)
# m.report(y)

# m.fit(X_train,y_train)
# # m.predict(X,y=y)
# # m.report(y)
# m.predict(X_test,y=y_test)
# m.predict_prob(X_test,y=y_test)
# m.report(y_test)
# # print(m.result['svc'])
# m.vote(y_test)