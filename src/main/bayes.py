import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
 
x=np.loadtxt("./data/wine.data" , delimiter = "," , usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13) )      #获取属性集
y=np.loadtxt("./data/wine.data" , delimiter = "," , usecols=(0) )  

#print(y)
#print(len(y))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)       
 
md = GaussianNB()
md.fit(x_train, y_train)

#print(md)
expected = y_test
predicted = md.predict(x_test)  #输出测试结果
#结果报告输出
print(metrics.classification_report(expected, predicted))    #输出结果，精确度、召回率、f-1分数
print(metrics.confusion_matrix(expected, predicted))         #混淆矩阵
