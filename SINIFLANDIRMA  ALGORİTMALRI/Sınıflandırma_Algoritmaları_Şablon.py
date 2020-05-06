# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:44:35 2020

@author: Enes Safa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

veriler = pd.read_csv('veriler.csv')
print(veriler)


x=veriler.iloc[:,1:4].values # bağımsız değişkenler
y=veriler.iloc[:,4:].values#bağımlı değişkenler



# eğitim verileri ayırma
from sklearn.model_selection import train_test_split

x_train, x_test, y_train , y_test = train_test_split(x,y,test_size=0.33, random_state=0 )

# öznitelik ölçeklendirme
from sklearn.preprocessing import StandardScaler

sc= StandardScaler()

X_train= sc.fit_transform(x_train) # x_train den öğren ve transform et
X_test= sc.transform(x_test)

# fit işlemi eğitme işlemidir  transform ise o eğitimi kullanma işlemidir


# Logistic regression
from sklearn.linear_model import LogisticRegression

logr=LogisticRegression(random_state=0)
logr.fit(X_train,y_train) # eğitim

y_pred= logr.predict(X_test) # tahmin
print("LOGR")
print(y_pred)
print(y_test)

#  Karmaşıklık Matrisi (Confusion)
cm=confusion_matrix(y_test,y_pred)
print(cm)

# KNN 
from sklearn.neighbors import KNeighborsClassifier 

knn=KNeighborsClassifier(n_neighbors=1, metric='minkowski') # Burada n değeri algoritmanın başarımı için önemli veri kümesine göre seçimesi gereklidir.
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
print('KNN')
cm=confusion_matrix(y_test,y_pred)
print(cm)


# SVC (SVM classifier)
from sklearn.svm import SVC
svc=SVC(kernel='rbf') # Farklı kernel deneyerek başarımı değiştire biliriz
svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)

# Naive Bayes ( koşullu olasılık)  gaussion yöntemi 
from sklearn.naive_bayes import GaussianNB
 
gnb=GaussianNB()
gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)
'''
# Naive Bayes ( koşullu olasılık)  multinominal yöntemi
from sklearn.naive_bayes import MultinomialNB

mnb=MultinomialNB()
mnb.fit(x_train, y_train)

y_pred=mnb.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print('MNB')
print(cm)

# Naive Bayes ( koşullu olasılık)  Bermoulli yöntemi
from sklearn.naive_bayes import BernoulliNB
bnb=BernoulliNB()
bnb.fit(X_test,y_test)

y_pred=bnb.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('BNB')
print(cm)

# Decision tree classifier(karar ağacı sınıflandırma)
'''# ID3 yötemini kullanmak için entropy seçtik
from sklearn.tree import DecisionTreeClassifier
dtc =DecisionTreeClassifier(criterion='entropy') # criterion(Bölmenin kalitesini ölçme işlevi) default değeri gini dir. Biz bunu entropy yaptık.Bilgi kazanımı için entropy seçtik
dtc.fit(X_train,y_train)

y_pred=dtc.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)

# Random Decision tree 
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=10, criterion='entropy')
rfc.fit(X_train,y_train)

y_pred=rfc.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)
print(y_test)


# ROC , TPR FPR değerleri
y_proba=rfc.predict_proba(X_test)  # sınıflandırma olasılıklarını göre bilmemiz için
print(y_proba[:,0]) # erkek olma durumu tahmin sütünu

from sklearn import metrics
fpr , tpr , thold= metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')# ilk parametre gerçek değerlerimiz sonraki parametre tahmin olasılıklarımız en sonuncucu sie pozitifolarak belirtmemizi istediği değişken
print(fpr)
print(tpr)

























 








































































