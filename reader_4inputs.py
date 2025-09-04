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

  kospi200 futures 65days-forward  predictions text files with variable input features,
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

  raw_df = pd.read_csv("kospi200f-943.csv", encoding="ISO-8859-1")
  raw_df = raw_df[:-config.predict_term]

  # column 0 : serial number 1 ~ ,  column 1: date
  test_start_index = np.int(raw_df[raw_df['date'] == config.test_start]['number'] - 1)

  input_target_list = [1]
  #input_target_list = list(range(1, 2 + config.input_size))
  #input_target_list = list(range(1, 39)) # 시고저종가, 수익률, 이동평균, . .
  #input_target_list = input_target_list + list(range(79, 147)) # 경제 관련 지수
  #input_target_list = input_target_list + list(range(195, 260)) # 통화량, 금리
  #input_target_list = input_target_list + list(range(260, 314)) # 수출입, 경상수지, 투자, 외화보유, 자산, 대외채무
  #input_target_list = input_target_list + list(range(314, 334)) # 환율
  #input_target_list = input_target_list + list(range(406, 454)) # 금리, 각국 국채금리
  #input_target_list = input_target_list + list(range(456, 561)) # 해외 주식, 상품, 원자재 지수
  #input_target_list = input_target_list + list(range(561, 565)) # 한국, 미국, 일본, 영국 경제 성장률
  #input_target_list = input_target_list + list(range(565, 934)) # 금융 자산 현황, 거래량
  input_target_list = input_target_list + list(range(934, 938)) # 시고저종가
  #input_target_list = input_target_list + list(range(938, 945)) # 이론가, 기준가, 기초자산가, 거래량(대금), 잔존일수, 미결제약정
  #input_target_list =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 195, 196, 197, 198, 199, 200, 207, 208, 211, 212, 213, 214, 215, 216, 217, 218, 220, 221, 222, 225, 227, 228, 229, 230, 231, 232, 233, 234, 235, 243, 247, 248, 250, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 277, 279, 280, 281, 282, 283, 284, 285, 286, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 316, 318, 320, 325, 326, 327, 329, 330, 331, 332, 334, 372, 386, 387, 389, 391, 393, 394, 395, 396, 400, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 474, 475, 476, 477, 478, 479, 480, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 505, 529, 537, 559, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944]

  if config.predict_term == 5: input_target_list.append(943+2)
  if config.predict_term == 20: input_target_list.append(943+3)
  if config.predict_term == 65: input_target_list.append(943+4)

  train_data =  raw_df.values[ : test_start_index, input_target_list]
  test_data = raw_df.values[test_start_index - config.step_interval * (config.num_steps-1) :, input_target_list]
  predict_data = test_data

  return train_data, test_data, predict_data, test_start_index

def make_index_date(test_data, predict_term, test_start_index, step_interval, num_steps):

  test_data_size = len(test_data) - step_interval * (num_steps-1)

  index_df = pd.read_csv("index-ma-std.csv", encoding = "ISO-8859-1")
  #예측 시점의 지수 list
  index = index_df.values[test_start_index: test_start_index + test_data_size, 1]
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