import pandas as pd
import numpy as np
from numpy import log,dot,exp,shape
from sklearn.datasets import make_classification

df = pd.read_csv('diabetes.csv')


"""print(len(df.loc[df['Glucose'] == 0]))
print(len(df.loc[df['BloodPressure'] == 0]))
print(len(df.loc[df['SkinThickness'] == 0]))
print(len(df.loc[df['Insulin'] == 0]))
print(len(df.loc[df['BMI'] == 0]))
print(len(df.loc[df['DiabetesPedigreeFunction'] == 0]))
print(len(df.loc[df['Age'] == 0]))"""

df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].mean())
df['BMI'] = df['BMI'].replace(0, df['BMI'].mean())

df = df.iloc[0: , :]

df = df.to_numpy().astype(np.single)

X1 = df[:,0:8]

y = df[:,8]



X1[:, 0] = (X1[:, 0] - X1[:, 0].min())/ (X1[:, 0].max() - X1[:, 0].min())

for i in range(8):
    data = (X1[:,i] - X1[:,i].min())/ (X1[:,i].max() - X1[:,i].min())
    X1[:,i] = data
    

arr1 = np.ones(768)
print(arr1.shape)

X1 = np.insert(X1, 0, arr1, axis=1)

print(X1[0:9,],y[0:9],)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

X1, y = unison_shuffled_copies(X1,y)


print("\n\n\n\npost shuffling")
print(X1[0:9,],y[0:9],)

