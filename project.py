import pandas as pd
import numpy as np
from numpy import log,dot,exp,shape
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score


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


X1 = np.insert(X1, 0, arr1, axis=1)



"""def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

X1, y = unison_shuffled_copies(X1,y)


print("\n\n\n\npost shuffling")"""

X_train, X_test, y_train, y_test = train_test_split(X1, y, train_size=0.8, random_state=0)
y_train = y_train[..., np.newaxis]
y_test = y_test[..., np.newaxis]

# print(shape(X_train))
# print(shape(X_test))
# print(shape(y_train))
# print(shape(y_test))
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

lr = 0.01
loss_list = []
total_loss = 0.0
theta_old = np.random.rand(9,1)
theta_new = None

for e in range(100):
    Z = np.matmul(X_train,theta_old)
    h = sigmoid(Z)
    
    J_vect = -y_train*log(h) - (1-y_train)*log(1-h) 

    J = np.mean(J_vect)
    total_loss += J
    loss_list.append(total_loss)
    grad = np.matmul(X_train.transpose(),(h-y_train))
    theta_new = theta_old - lr * grad
    theta_old = theta_new


Z = np.matmul(X_test, theta_old)
h = sigmoid(Z)
h[h >= 0.5] = 1 
h[h < 0.5] = 0


y_test_predicted = h
print(accuracy_score(y_test, y_test_predicted))

Z = np.matmul(X_train, theta_old)
h = sigmoid(Z)
h[h >= 0.5] = 1 
h[h < 0.5] = 0


y_train_predicted = h
print(accuracy_score(y_train, y_train_predicted))     
        