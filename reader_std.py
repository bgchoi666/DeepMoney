# Copyright 2018 Bimghi Choi. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf
import numpy as np
import pandas as pd


def raw_data(config, data_path=None):
  """Load PTB raw data from data directory "data_path".

  kospi200 futures 65days-forward  predictions text files with 35 input features,
  and performs mini-batching of the inputs.

  Args:
    config : configuration class
    data_path: not use

  Returns:
    tuple (train_data, valid_data, test_data)
    where each of the data objects can be passed to producer function.
  """

  #if config.predict_term == 5: raw_df = pd.read_csv("kospi200f-943-1week-norm.csv", encoding = "ISO-8859-1")
  #if config.predict_term == 20: raw_df = pd.read_csv("kospi200f-943-1month-norm.csv", encoding = "ISO-8859-1")
  #if config.predict_term == 65: raw_df = pd.read_csv("kospi200f-943-3months-norm.csv", encoding = "ISO-8859-1")

  raw_df = pd.read_csv("kospi200f-943-std.csv", encoding="ISO-8859-1")
  raw_df = raw_df[ : -config.predict_term]

  # column 0 : serial number 1 ~ ,  column 1: date
  test_start_index = np.int(raw_df[raw_df['date'] == config.test_start]['number'] - 1)

  input_target_list = list(range(1, 2 + config.input_size))

  #if config.predict_term == 5: input_target_list.append(config.input_size+2)
  #if config.predict_term == 20: input_target_list.append(config.input_size+3)
  #if config.predict_term == 65: input_target_list.append(config.input_size+4)

  input_target_list.append(config.input_size + 2)

  train_data =  raw_df.values[ : test_start_index, input_target_list]
  test_data = raw_df.values[test_start_index - config.step_interval * (config.num_steps-1) :, input_target_list]
  predict_data = test_data

  return train_data, test_data, predict_data, test_start_index

def make_index_date(test_data, predict_term, test_start_index, step_interval, num_steps):

  test_data_size = len(test_data) - step_interval * (num_steps-1)

  index_df = pd.read_csv("index-ma-std.csv", encoding = "ISO-8859-1")
  #예측 시점의 std list
  if predict_term == 5:
    index = index_df.values[test_start_index: test_start_index + test_data_size, 5]
  if predict_term == 20:
    index = index_df.values[test_start_index: test_start_index + test_data_size, 6]
  if predict_term == 65:
    index = index_df.values[test_start_index: test_start_index + test_data_size, 7]
  #예측 대상일 list
  date = index_df.values[test_start_index + predict_term: test_start_index +  predict_term + test_data_size, 0]
  #trade_date data
  #예측하는 날짜 list
  z = index_df.values[test_start_index : -predict_term, 0]
  z = list(z)

  return index, date, z

def producer(raw_data, num_steps, step_interval, input_size, output_size, name=None):
  """produce time-series data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from futures data.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the target data(65days-forward kospi200 futures values).

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """

  # nput과 target 분리하여 반환
  dataX, dataY = [], []

  # series 데이터 건수 계산
  size = len(raw_data) - (num_steps-1) * step_interval

  for i in range(size):
    input_list = list(range(i, i + num_steps * step_interval, step_interval))
    a = np.float32(raw_data[input_list, 1:input_size+1])
    dataX.append(a)
    b = np.float32(raw_data[input_list, input_size+1])
    dataY.append(b)

  # 3차원 array로 dimension 조정
  x = np.array(dataX).reshape(size, num_steps, input_size)
  y = np.array(dataY).reshape(size, num_steps, output_size)

  return x, y