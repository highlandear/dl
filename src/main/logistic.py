import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import LogisticRegression

from sklearn import metrics

x=np.loadtxt("./data/wine.data" , delimiter = "," , usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13) )      #获取属性集
y=np.loadtxt("./data/wine.data" , delimiter = "," , usecols=(0) )  

#print(y)
#print(len(y))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) 

#print(len(y_train))
#print(y_train)


md = LogisticRegression()
md.fit(x_train, y_train)

#print(md)
expected = y_test
predicted = md.predict(x_test)

a = []
b = []
for v in expected:
    a.append(v)
for v in predicted:
    b.append(v)
print('=============================')
print(a.count(1),a.count(2),a.count(3))
print(b.count(1),b.count(2),b.count(3))

print(metrics.classification_report(expected, predicted))

#print(metrics.confusion_matrix(expected, predicted))            #混淆矩阵