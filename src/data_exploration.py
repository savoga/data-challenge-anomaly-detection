import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import random as random
import time
import seaborn as sns
from sklearn.decomposition import PCA

# load data
data = np.loadtxt('airbus_train.csv', delimiter= ' ', max_rows=100) # remove 100
data_test = np.loadtxt('airbus_test.csv', delimiter= ' ', max_rows=100) # remove 100

ax = plt.figure(figsize=(40,5)).gca()
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# Display 5 first entries
for i in range(5):
    plt.plot(range(1,6),data[i,:5])
plt.title('Observations for the 5 first entries')

# display random entries
ax = plt.figure(figsize=(40,5)).gca()
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

for i in range(5):
    plt.plot(range(1,6),data[i,:5])
plt.title('Observations for the 5 first entries')

# display random test data
start_time = time.time()
random_state_list = [59,46,2,1,899]
for s in random_state_list:
  random.seed(s)
  n = int(random.random()*data_test.shape[0])
  print(n)
  fig = plt.figure(figsize=(40,5))
  plt.plot(range(data.shape[1]),data_test[n,:])
print("{} seconds".format(round(time.time() - start_time,2)))

# Standardization
data_df = pd.DataFrame(data)
data_sample = data_df.sample(n=4, random_state=11, axis=1)
sns.pairplot(data_sample, size=2.5)

data_df = pd.DataFrame(data_test)
data_sample = data_df.sample(n=4, random_state=11, axis=1)
sns.pairplot(data_sample, size=2.5)

data_df.describe() # all stats

# 2 dimensions display using PCA
pca = PCA(n_components = 2, whiten = True)
data_reduced = pca.fit_transform(data)

plt.plot(data_reduced[:,0],data_reduced[:,1], '+')
plt.title('PCA on train set (2 components)')

pca_test_2 = PCA(n_components = 2, whiten = True)
data_test_reduced = pca_test_2.fit_transform(data_test)

plt.plot(data_test_reduced[:,0],data_test_reduced[:,1], '+')
plt.title('PCA on test set (2 components)')

# Display most important components
pca_100 = PCA(n_components = 100)
data_reduced_full = pca_100.fit_transform(data)
plt.plot(range(100),pca_100.explained_variance_ratio_)