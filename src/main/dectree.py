# https://blog.csdn.net/u014157632/article/details/59181772
import pandas as pd
 
data = pd.read_excel("./data/decision.xlsx", index_col = 0) 
data[data == u'好'] = 1
data[data == u'是'] = 1
data[data == u'高'] = 1
data[data != 1] = -1
#print(data)
x = data.iloc[:,:3].as_matrix().astype(int)
y = data.iloc[:,3].as_matrix().astype(int)
print(x)
print(y)