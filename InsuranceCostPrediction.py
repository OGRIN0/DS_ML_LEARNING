import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sagemaker.workflow.airflow import training_config
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from GridSearch import x_train, x_test, y_train, y_test

df = pd.read_csv('/Users/amarnath/DS_ML_learning/src/insurance.csv')

sns.set()
plt.figure(figsize=(6,6))
sns.displot(df['age'])
plt.title("Age Distribution")
plt.show() #distribution plot fot age

sns.countplot(x="sex", data=df)
plt.title("Sex Distribution")
plt.show() #distribution for sex

df['sex'].value_counts()

sns.displot(df['bmi'])

plt.show() #distribution for bmi

df.replace({'sex':{"male":0, 'female':1}}, inplace=True)
df.replace({'smoker':{'yes':0, 'no':1}}, inplace=True)
df.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)

x = df.drop(columns="charges", axis=1)
y = df['charges']

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

z = LinearRegression()

z.fit(x_train, y_train)

training_data_pred = z.predict(x_train)

r2_train= metrics.r2_score(y_train, training_data_pred)

print(r2_train)#prints accuracy value for training data prediction

test_data_pred = z.predict(x_test)
a = (metrics.r2_score(y_test, test_data_pred))
print(a) #prints test data prediction

sd = (30,1,22.7,0,1,0)#sample data for testing model

inp_narray = np.asarray(sd)

inp_re = inp_narray.reshape(1,-1)

pred = z.predict(inp_re)

print("Insurance Cost for sample data:", pred)


