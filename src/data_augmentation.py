import numpy as np
import tsaug

data = np.loadtxt('airbus_train.csv', delimiter= ' ', max_rows=100) # remove 100
data_test = np.loadtxt('airbus_test.csv', delimiter= ' ', max_rows=100) # remove 100

'''
Moving average
'''

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

data_ma = np.zeros((data.shape[0],data.shape[1]-4))
for i in range(data.shape[0]):
  data_ma[i] = moving_average(data[i],5)
data_extend = np.concatenate((data[:,4:],data_ma), axis=0)

'''
Random noise
'''
mu, sigma = 0, 0.1
noise = np.random.normal(mu, sigma, [data.shape[0],data.shape[1]])
data_noise = data + noise
data_extend = np.concatenate((data, data_noise), axis=0)

'''
Time warping
'''
data_time = np.zeros((500,data.shape[1]))
for i in range(500):
  data_time[i]=tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=2).augment(data[i])
data_extend_time = np.concatenate((data_extend, data_time), axis=0)