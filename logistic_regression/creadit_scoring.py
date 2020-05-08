import numpy as np
import pandas as pd
import zipfile
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

pd.set_option('display.max_columns', 500)
with zipfile.ZipFile('KaggleCredit2.csv.zip', 'r') as z:  # 读取zip里的文件
    f = z.open('KaggleCredit2.csv')
    data = pd.read_csv(f, index_col=0)
print(data.head())

print(data.shape)

data.isnull().sum(axis=0)

data.dropna(inplace=True)  # 去掉为空的数据
print(data.shape)

y = data['SeriousDlqin2yrs']
X = data.drop('SeriousDlqin2yrs', axis=1)

print(y.mean())  # 求取均值

x_tran,x_test,y_tran,y_test=model_selection.train_test_split(X,y,test_size=0.2)
print(x_test.shape)

## https://blog.csdn.net/sun_shengyun/article/details/53811483
lr = LogisticRegression(multi_class='ovr', solver='sag',
                        class_weight='balanced')
lr.fit(x_tran, y_tran)
score = lr.score(x_tran, y_tran)
print(score)  # 最好的分数是1

## https://blog.csdn.net/qq_16095417/article/details/79590455
train_score = accuracy_score(y_tran, lr.predict(x_tran))
test_score = lr.score(x_test, y_test)
print('训练集准确率：', train_score)
print('测试集准确率：', test_score)

##召回率
train_recall = recall_score(y_tran, lr.predict(x_tran), average='macro')
test_recall = recall_score(y_test, lr.predict(x_test), average='macro')
print('训练集召回率：', train_recall)
print('测试集召回率：', test_recall)

y_pro = lr.predict_proba(x_test)  # 获取预测概率值
y_prd2 = [list(p >= 0.3).index(1)
          for i, p in enumerate(y_pro)]  # 设定0.3阈值，把大于0.3的看成1分类。
train_score = accuracy_score(y_test, y_prd2)
print(train_score)
