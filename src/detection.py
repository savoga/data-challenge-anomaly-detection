import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.stattools import adfuller
import tensorflow as tf
print(tf.__version__)

sns.set(color_codes=True)

from numpy.random import seed

from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model, Sequential
from keras import regularizers
from keras.initializers import RandomNormal

from sklearn.ensemble import IsolationForest

from tsfresh.feature_extraction import feature_calculators as tsf_calc

data = np.loadtxt('airbus_train.csv', delimiter= ' ', max_rows=100) # remove 100
data_test = np.loadtxt('airbus_test.csv', delimiter= ' ', max_rows=100) # remove 100

ax = plt.figure(figsize=(40,5)).gca()
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

'''
PCA + LOF(20)
'''

pca_20 = PCA(n_components = 20)
data_reduced_20 = pca_20.fit_transform(data)

lof = LocalOutlierFactor(n_neighbors = 5, novelty = True, metric= 'euclidean')
lof.fit(data_reduced_20)

data_test_reduced = pca_20.fit_transform(data_test)

scores = -lof.score_samples(data_test_reduced) # we take the opposite since we want the higher the more abnormal

# np.savetxt('/content/gdrive/My Drive/scores_anomaly_pca_lof.csv', scores, fmt = '%1.6f', delimiter=',')

'''
Average + ADF
'''

a = np.zeros((data_test.shape[0], int(data_test.shape[1]/1024)))
s = 0
for i in range(data_test.shape[0]):
  k=0
  for j in range(data_test.shape[1]):
    s += data_test[i,j]
    if(j % 1024 != 0):
      continue
    else:
      a[i,k] = s / 60
      k += 1
      s = 0

res = []
for i in range(data_test.shape[0]):
  res_af = adfuller(a[i], autolag='AIC')
  res.append(res_af[1])

# np.savetxt('/content/gdrive/My Drive/scores_adf.csv', np.array(res), fmt = '%1.6f', delimiter=',')

'''
Autoencoder
'''

seed(10)
tf.random.set_seed(10)
act_func = 'relu'

model=Sequential()

model.add(Dense(25,activation=act_func,
                kernel_initializer=RandomNormal(),
                kernel_regularizer=regularizers.l2(0.0),
                input_shape=(data.shape[1],)
               )
         )
model.add(Dense(3,activation=act_func,
                kernel_initializer=RandomNormal()))
model.add(Dense(25,activation=act_func,
                kernel_initializer=RandomNormal()))
model.add(Dense(data.shape[1],
                kernel_initializer=RandomNormal()))
model.compile(loss='mse',optimizer='adam')

NUM_EPOCHS=5
BATCH_SIZE=10

history=model.fit(np.array(data),np.array(data),
                  batch_size=BATCH_SIZE,
                  epochs=NUM_EPOCHS,
                  validation_split=0.05,
                  verbose = 1)


X_pred_test = model.predict(data_test)

scores_test = np.mean(np.abs(X_pred_test-data_test), axis = 1)

# np.savetxt('/content/gdrive/My Drive/scores_ae.csv', scores_test, fmt = '%1.6f', delimiter=',')

'''
Score averaging
'''

scores_pca = np.loadtxt('scores_anomaly_pca_lof.csv', delimiter= ' ')
scores_ae = np.loadtxt('scores_anomaly_autoencoder.csv', delimiter= ' ')

ave_scores = []
for i in range(len(scores_pca)):
  ave_scores.append((scores_pca[i] + scores_ae[i]) /2)

# np.savetxt('/content/gdrive/My Drive/scores_ave_pca_ae.csv', ave_scores, fmt = '%1.6f', delimiter=',')

'''
PCA + IF
'''

pca_5 = PCA(n_components = 5)
data_reduced_5 = pca_5.fit_transform(data)

clf_if = IsolationForest(n_estimators=30, max_samples=100, random_state=0)
clf_if.fit(data_reduced_5)

data_test_reduced = pca_5.fit_transform(data_test)

scores_if = clf_if.decision_function(data_test_reduced)

# np.savetxt('/content/gdrive/My Drive/scores_if_pca.csv', -scores_if, fmt = '%1.6f', delimiter=',')

'''
Autoencoder + IF
'''

encoding_dim = 20

input_df = Input(shape=(data.shape[1],))
encoded = Dense(encoding_dim, activation='relu')(input_df)
decoded = Dense(data.shape[1], activation='sigmoid')(encoded)

autoencoder = Model(input_df, decoded)

encoder = Model(input_df, encoded)

autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

NUM_EPOCHS=500
BATCH_SIZE=10

autoencoder.fit(np.array(data),np.array(data),
                  batch_size=BATCH_SIZE,
                  epochs=NUM_EPOCHS,
                  validation_split=0.05,
                  verbose = 1)

encoded_data = encoder.predict(data)

clf_if = IsolationForest(n_estimators=30, max_samples=100, random_state=0)
clf_if.fit(encoded_data)

encoded_data_test = encoder.predict(data_test)

scores_if = clf_if.decision_function(encoded_data_test)

# np.savetxt('/content/gdrive/My Drive/scores_autoencoder_if.csv', scores_if, fmt = '%1.6f', delimiter=',')

'''
Feature engineering + LOF
'''

def standardize(s):
  return (s - np.mean(s))/np.std(s)

def preproc(d):
  df = pd.DataFrame(d)
  x_autocorr = df.apply(lambda x: x.autocorr(lag=5), axis=1)
  x_mean = df.apply(lambda x: np.mean(x), axis=1)
  x_max = df.apply(lambda x: np.max(x), axis=1)
  x_c3 = df.apply(lambda x: tsf_calc.c3(x,5), axis=1)
  x_cid = standardize(df.apply(lambda x: tsf_calc.cid_ce(x,False), axis=1))
  x_sym = df.apply(lambda x: 0 if tsf_calc.symmetry_looking(x,[{'r':0.0106}])[0][1] else 1, axis=1) # all observations that are strongly asymmetric

  return pd.concat([x_autocorr, x_mean, x_max, x_c3, x_cid, x_sym], axis=1)

data_prep = preproc(data)
data_test_prep = preproc(data_test).fillna(500)

lof = LocalOutlierFactor(n_neighbors = 7, novelty = True, metric= 'euclidean')
lof.fit(data_prep)
scores = -lof.score_samples(data_test_prep)

# np.savetxt('/content/gdrive/My Drive/scores_long4000_std.csv', scores, fmt = '%1.6f', delimiter=',')