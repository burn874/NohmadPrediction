from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import os
import tensorflow as tf
import itertools
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest

TRAIN_FALSE = 'bandgap_energy_ev'
TRAIN_TRUE = 'formation_energy_ev_natom'

# Import and init tensorflow
tf.logging.set_verbosity(tf.logging.INFO)
sess = tf.InteractiveSession()

train = pd.read_csv('C:\\Users\\user\\Desktop\\cz4041\\data_file\\train.csv')
print('Shape of the train data with all features:', train.shape)
train = train.select_dtypes(exclude = ['object'])

print('Shape of the train data with numerical features:', train.shape)
train.drop('id', axis = 1, inplace = True)
train.drop(TRAIN_FALSE, axis = 1, inplace = True)
train.fillna(0, inplace = True)

test = pd.read_csv('C:\\Users\\user\\Desktop\\cz4041\\data_file\\test.csv')
test = test.select_dtypes(exclude=['object'])
ID = test.id
test.fillna(0, inplace = True)
test.drop('id', axis = 1, inplace = True)

print("List of features contained our dataset:", list(train.columns))

# Outliers
clf = IsolationForest(max_samples = 100, random_state = 42)
clf.fit(train)
y_noano = clf.predict(train)
y_noano = pd.DataFrame(y_noano, columns = ['Top'])
y_noano[y_noano['Top'] == 1].index.values

train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]
train.reset_index(drop = True, inplace = True)
print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
print("Number of rows without outliers:", train.shape[0])


# Preprocessing
warnings.filterwarnings('ignore')

col_train = list(train.columns)
col_train_bis = list(train.columns)

col_train_bis.remove(TRAIN_TRUE)

mat_train = np.matrix(train)
mat_test  = np.matrix(test)
mat_new = np.matrix(train.drop(TRAIN_TRUE, axis = 1))
mat_y = np.array(train.formation_energy_ev_natom).reshape((2160,1))

prepro_y = MinMaxScaler()
prepro_y.fit(mat_y)

prepro = MinMaxScaler()
prepro.fit(mat_train)

prepro_test = MinMaxScaler()
prepro_test.fit(mat_new)

train = pd.DataFrame(prepro.transform(mat_train),columns = col_train)
test = pd.DataFrame(prepro_test.transform(mat_test),columns = col_train_bis)

print(len(test))
train.head()

# List of features
COLUMNS = col_train
FEATURES = col_train_bis
LABEL = TRAIN_TRUE

# Columns for tensorflow
feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

# Training set and Prediction set with the features to predict
training_set = train[COLUMNS]
prediction_set = train.formation_energy_ev_natom

# Train and Test
x_train, x_test, y_train, y_test = train_test_split(training_set[FEATURES] , prediction_set, test_size = 0.2, random_state = 42)
y_train = pd.DataFrame(y_train, columns = [LABEL])
training_set = pd.DataFrame(x_train, columns = FEATURES).merge(y_train, left_index = True, right_index = True)
training_set.head()

# Training for submission
training_sub = training_set[col_train]

# Same thing but for the test set
y_test = pd.DataFrame(y_test, columns = [LABEL])

testing_set = pd.DataFrame(x_test, columns = FEATURES).merge(y_test, left_index = True, right_index = True)
testing_set.head()


# Start of Model
# Model
tf.logging.set_verbosity(tf.logging.ERROR)
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                          activation_fn = tf.nn.relu, hidden_units=[200, 100, 50, 25, 12])

# Reset the index of training
training_set.reset_index(drop = True, inplace =True)


def input_fn(data_set, pred=False):
   if pred == False:
      feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
      labels = tf.constant(data_set[LABEL].values)

      return feature_cols, labels

   if pred == True:
      feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}

      return feature_cols


# Deep Neural Network Regressor with the training set which contain the data split by train test split
regressor.fit(input_fn = lambda: input_fn(training_set), steps = 2000)

# Evaluation on the test set created by train_test_split
ev = regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)

# Display the score on the testing set
# 0.002X in average
loss_score1 = ev["loss"]
print("Final Loss on the testing set: {0:f}".format(loss_score1))

# Test
y_predict = regressor.predict(input_fn = lambda: input_fn(test, pred = True))

def to_submit(pred_y, name_out):
   y_predict = list(itertools.islice(pred_y, test.shape[0]))
   y_predict = pd.DataFrame(prepro_y.inverse_transform(np.array(y_predict).reshape(len(y_predict), 1)),
                            columns=[TRAIN_TRUE])
   y_predict = y_predict.join(ID)
   y_predict.to_csv(name_out + '.csv', index = False)


to_submit(y_predict, "submission")