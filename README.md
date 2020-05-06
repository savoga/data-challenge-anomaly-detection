<p align="center"><img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/airbus_helicopters.png"></img></p>

# Data Challenge - Anomaly Detection

This is a university project in the form of a data challenge that I did during my data science degree at Télécom Paris.

## Context

The data set is provided by Airbus and consists of the measures of the accelerometer of helicopters during 1 minute at frequency 1024 Hertz, which yields time series measured at in total 60 * 1024 = 61440 equidistant time points.

## Data

<p align="center"><img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/data_structure.png"></img></p>

### Training data

The training set consists of one file, airbus_train.csv.

File airbus_train.csv contains one observation per row, each observation having 61440 entries, measures with equivalent time distance of 1 / 1024 seconds.

There are in total 1677 training observations.

### Test data

The training set consists of one file, airbus_test.csv, which has the same structure as file airbus_train.csv.

There are in total 2511 test observations.

## Introduction

For this challenge, I've chosen first to perform some basic analysis to understand the data. This analysis consists of displaying train/test data in order to spot the differences. This allowed me to draw conclusions on the difference of both datasets e.g. the presence of outliers, standardization, stationarity,... and to have a first idea of interesting features to use for the future of my work.

Then, I tried a large number of different algorithms. It turns out that simple Feature Engineering worked the best. I thus focused on this method using the best features as possible. Important to note that the package *tsfresh* brought significant value to my research.

## 1. Data exploration

#### Autocorrelation

5 first entries (columns):

<p align="center"><img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/first_entries.png"></img></p>

5 random entries:

<p align="center"><img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/random_entries.png"></img></p>

Since random entries give much more volatile trends, our first conclusion is that autocorrelation may be a good feature to consider (which is often the case with time series).

#### Standardization

Looking at standardization is important to make sure all features contribute equally to the analysis.

After sampling few data on both sets, we can plot few distributions:

<p align="center"><img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/standardization_all.png"></img></p>

Graphs from train set (left) indicates that data are standardized.
We can draw the same conclusion for the graph of the test set, however a possible presence of outliers give more spread values (higher variance). This will be confirmed in the next section.

#### Outlier presence

The PCA allows us to display observations in 2 dimensions. We can thus easily spot outliers on the test set (right).

<p align="center"><img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/pca_all.png"></img></p>

#### Stationarity

2 random observations from the train set:

<p align="center"><img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/stationarity_train.png"></img></p>

We can see that the both series seem quite stationary. It's less obvious for random observations for the test set:

<p align="center"><img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/stationarity_test.png"></img></p>

Although the first observation can seem stationary, the mean is quite high compared to most of the observations (as seen previously) which makes it an outlier.

## 2. Anomaly detection


