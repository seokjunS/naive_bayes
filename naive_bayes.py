"""
" An implementation of Naive Bayes for multiple categorical features.
" Create at: 2017-09-17
" Author: seokjunS
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from sklearn import model_selection, naive_bayes, metrics, preprocessing




class NaiveBayes(object):
  """
  " Naive Bayes model
  " Traning model with 'fit' function, while predicting new instance with 'predict' function.
  " 'predict_proba' function returns probability for each target classes.
  """
  def __init__(self, alpha = 0.0):
    self.alpha = alpha

    self.data_encoders_ = []
    self.target_encoder_ = None

    self.class_count_ = None
    self.class_log_prior_ = []
    self.feature_log_prob_ = []

  def fit(self, data, target):
    # copy data
    data = deepcopy(data)
    target = deepcopy(target)

    # encode target class labels
    encoded, enc = self.learn_encoding( target )
    target = encoded
    self.target_encoder_ = enc

    # calculate prior of target classes by counting
    class_counts = self.counting(target)
    self.class_log_prior_ = np.log( class_counts / sum(class_counts) )

    # encode features and calculate likelihood (log probability) by counting
    for colname in data:
      encoded, enc = self.learn_encoding( data[colname] )
      data[colname] = encoded
      self.data_encoders_.append( enc )

      # add alpha smoothing
      prob = np.full((len(enc.classes_), len(self.class_log_prior_)), fill_value=self.alpha)
      
      # counting features
      for i, v in enumerate(data[colname]):
        t = target[i]
        prob[v,t] += 1.0

      # convert counts to probability
      prob = prob / np.sum(prob, axis=0)

      # set log probability
      self.feature_log_prob_.append( np.log(prob) )


  def predict_proba(self, data):
    res = np.zeros((data.shape[0], len(self.class_log_prior_)))
    data = deepcopy(data)

    # encode features
    for i, colname in enumerate(data):
      data[colname] = self.encoding( data[colname], self.data_encoders_[i] )

    # calculate unnormalized posterior probability
    for di, row in data.iterrows():
      prob = 0.0
      for i, fp in enumerate(self.feature_log_prob_):
        prob += fp[ row[i] ]
      prob += self.class_log_prior_
      res[di] = np.exp(prob)
    
    # return normalized posterior probability
    return res / np.sum(res)


  def predict(self, data):
    # calculate probability for each class
    pred = self.predict_proba(data)
    # select index whose probability is maximum.
    labels = np.argmax(pred, axis=1)
    # convert index to target class label
    return self.target_encoder_.inverse_transform(labels)


  ### helpers
  def learn_encoding(self, col):
    enc = preprocessing.LabelEncoder()
    encoded = enc.fit_transform( col )
    return encoded, enc

  def encoding(self, col, enc):
    return enc.transform( col )

  def counting(self, arr):
    # counting all different values and return counts in numerical order of original values
    cnts = defaultdict(int)

    for v in arr:
      cnts[v] += 1.0

    return np.array([ v for k,v in sorted(cnts.items(), key=lambda x: x[0]) ])







if __name__ == '__main__':
  ### define input data
  # Load play tennis whose first row is data header
  rawdata = pd.read_csv('play_tennis.csv')

  data = rawdata.iloc[:,:-1]
  target = rawdata.iloc[:,-1]

  model = NaiveBayes(alpha=0.0)

  model.fit(data, target)

  test_data = np.array([['sunny', 'cool', 'high', 'strong']])

  test_data = pd.DataFrame(test_data, columns=['outlook', 'temp', 'humidity', 'wind'])
  print(test_data)
  print(model.predict_proba(test_data))
  print(model.predict(test_data))



