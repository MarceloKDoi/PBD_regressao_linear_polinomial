import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('HeightVsWeight.csv')
# print(dataset)
ind = dataset.iloc[:, -2:-1].values
dep = dataset.iloc[:, -1].values

# print("ind:\n", ind)
# print("dep:\n", dep)

dep = dep.reshape(len(dep), 1)

# print("Depois -ind:\n", ind)
# print("Depois -dep:\n", dep)

indScaler = StandardScaler()
depScaler = StandardScaler()
ind = indScaler.fit_transform(ind)
dep = depScaler.fit_transform(dep)


# degree

svr = SVR(kernel='rbf', degree=4)
svr.fit(ind, dep)


plt.scatter(ind, dep, color="red")

plt.plot(ind, svr.predict(ind))

plt.title("Regressão por Vetor de Suporte")
plt.xlabel("Nível")
plt.ylabel("Salário")
plt.show()
