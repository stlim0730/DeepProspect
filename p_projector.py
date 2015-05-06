from __future__ import division
import pandas as pd
import numpy as np
from numpy import nan
import scipy as sp
import math
from sklearn.svm import SVC
import sys
import fpformat
import random
import copy
import pickle
from my_util import *

MINOR_FIELDS = ['PID', 'NAME', 'AGE', 'T', 'YR', 'TM', 'G', 'GS', 'W', 'L', 'CG', 'ShO', 'IP', 'H', 'R', 'ER', 'TBB', 'IBB', 'SO', 'ERA', 'GF', 'SV', 'HR', 'HB', 'BK', 'WP', 'SO/BB', 'H/9', 'SO/9', 'BB/9', 'HR/9', 'BR/9']
MINOR_COPY = ['PID', 'NAME']
MINOR_PER_YEAR = ['G', 'GS', 'W', 'L', 'CG', 'ShO', 'H', 'R', 'ER', 'TBB', 'IBB', 'SO', 'GF', 'SV', 'HR', 'HB', 'BK', 'WP', 'H/9', 'SO/9', 'BB/9', 'HR/9', 'BR/9']

MAJOR_FIELDS = ['PID', 'NAME', 'AGE', 'T', 'YR', 'TM', 'G', 'GS', 'W', 'L', 'CG', 'ShO', 'IP', 'H', 'R', 'ER', 'TBB', 'IBB', 'SO', 'ERA', 'GF', 'SV', 'HR', 'HB', 'BK', 'WP', 'SO/BB', 'H/9', 'SO/9', 'BB/9', 'HR/9', 'BR/9', 'RAA', 'WAA', 'WAAadj', 'WAR', 'RAR', 'waaWL%', '162WL%']
MAJOR_COPY = ['PID', 'NAME']
MAJOR_PER_YEAR = ['G', 'GS', 'W', 'L', 'CG', 'ShO', 'H', 'R', 'ER', 'TBB', 'IBB', 'SO', 'GF', 'SV', 'HR', 'HB', 'BK', 'WP', 'H/9', 'SO/9', 'BB/9', 'HR/9', 'BR/9', 'RAA', 'WAA', 'WAAadj', 'WAR', 'RAR', 'waaWL%', '162WL%']

# DEFAULT_PLAYER_OBJ = {
#   'PID': '', 'NAME': '', 'AGE': 0, 'G': 0, 'GS': 0, 'W': 0, 'L': 0, 'WLP': 0, 'CG': 0, 'ShO': 0, 'IP': 0, 'H': 0, 'R': 0, 'ER': 0, 'TBB': 0, 'IBB': 0, 'SO': 0, 'ERA': 0, 'GF': 0, 'SV': 0, 'HR': 0, 'HB': 0, 'BK': 0, 'WP': 0, 'SO/BB': 0, 'H/9': 0, 'SO/9': 0, 'BB/9': 0, 'HR/9': 0, 'BR/9': 0, 'RAA': 0, 'WAA': 0, 'WAAadj': 0, 'WAR': 0, 'RAR': 0, 'waaWL%': 0, '162WL%': 0
# }

DEFAULT_AGGR_RECORD = {
  'PID': '', 'NAME': '',
  'G': 0, 'GS': 0, 'W': 0, 'L': 0,
  'CG': 0, 'ShO': 0, 'IP': 0,
  'H': 0, 'R': 0, 'ER': 0,
  'TBB': 0, 'IBB': 0, 'SO': 0, 'ERA': 0,
  'GF': 0, 'SV': 0,
  'HR': 0, 'HB': 0, 'BK': 0, 'WP': 0,
  'H/9': 0, 'SO/9': 0, 'BB/9': 0, 'HR/9': 0, 'BR/9': 0,
  'WAR': -162
}

MINOR_YEARS = 3

FILE_NAME_PREFIX = 'PitcherData_BR_'
CSV = '.csv'
MINOR_SUFFIX = '-min'
# there shouldn't be an 'x'!
LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z']

KERNEL_TYPES = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']

skip_cnt = 0;

def is_minor_record(row):
  team = row['TM']
  if team.endswith(MINOR_SUFFIX):
    return True
  else:
    return False

def sum_innings(innings_list):
  innings_int = []
  innings_flt = []
  for inning in innings_list:
    inning_int = math.floor(inning)
    innings_int.append(inning_int)
    inning_flt = round(inning - inning_int, 1)
    innings_flt.append(inning_flt)
  innings_flt = sorted(innings_flt, key=float)
  flt_sum = 0.0
  carry = 0
  for i in range(len(innings_flt)):
    if flt_sum == 0.0:
      flt_sum = innings_flt[i]
    elif flt_sum == 0.1 and innings_flt[i] == 0.1:
      flt_sum = 0.2
    elif flt_sum == 0.1 and innings_flt[i] == 0.2:
      flt_sum = 0.0
      carry += 1
    elif flt_sum == 0.2 and innings_flt[i] == 0.2:
      flt_sum = 0.1
      carry += 1
    elif flt_sum == 0.2 and innings_flt[i] == 0.1:
      flt_sum = 0.0
      carry += 1
  return sum(innings_int) + carry + flt_sum

def convert_inning_floats(inning):
  inning_int = math.floor(inning)
  inning_flt = round(inning - inning_int, 1) * 0.33
  return inning_int + inning_flt

def getColumnList(dataFrame, columnName):
  return dataFrame[columnName].tolist()
  
def getTrainedSVM(inputList, labelList, verbose=False, kern='rbf', C=1.0, gamma=0.0):
  classifier = SVC(kernel=kern, C=C, gamma=gamma)
  fitRes = classifier.fit(inputList, labelList)
  if verbose:
    print fitRes
  return classifier

def shuffle_df(df):
  df_list = []
  for index, row in df.iterrows():
    df_list.append(row)
  np.random.shuffle(df_list)
  return pd.DataFrame.from_records(df_list)

def aggregate(player_df, fields_copy, fields_per_year, years = MINOR_YEARS):
  agg_record = copy.deepcopy(DEFAULT_AGGR_RECORD)
  young_career_df = player_df.iloc[:years]
  num_years = len(young_career_df)
  for field in fields_copy:
    first_row = young_career_df.iloc[0]
    agg_record[field] = first_row[field]
  for field in fields_per_year:
    sum_list = young_career_df[field].tolist()
    agg_record[field] = sum(sum_list) / num_years
  ip_list = young_career_df['IP'].tolist()
  agg_record['IP'] = sum_innings(ip_list)
  inning_pitched = convert_inning_floats(sum_innings(ip_list))
  if agg_record['IP'] == 0.0 and agg_record['ER'] != 0:
    agg_record['ERA'] = 99.99
  elif agg_record['IP'] == 0:
    agg_record['ERA'] == 0.00
  else:
    agg_record['ERA'] = round((agg_record['ER'] / inning_pitched) * 9, 2)
  return agg_record

# def aggregate(row_list, fields_sum, fields_avg, fields_copy, is_minor = False):
#   result = DEFAULT_PLAYER_OBJ
#   for field in fields_sum:
#     field_sum = 0
#     for row in row_list:
#       field_sum += row[field]
#     result[field] = field_sum
#   for field in fields_avg:
#     field_sum = 0
#     for row in row_list:
#       field_sum += row[field]
#     result[field] = field_sum / len(row_list)
#   # for field in fields_copy:
#   #   result[field] = row_list[0][field]
#   # result['WLP'] = round(result['W'] / (result['W'] + result['L']), 3)
#   ip_list = []
#   for row in row_list:
#     ip_list.append(row['IP'])
#   result['IP'] = sum_innings(ip_list)
#   if result['IP'] == 0 and result['ER'] != 0:
#     result['ERA'] = 99.99
#   elif result['IP'] == 0:
#     result['ERA'] == 0.00
#   else:
#     result['ERA'] = round((result['ER'] / result['IP']) * 9, 2)
#   # TODO: 'SO/BB', 'H/9', 'SO/9', 'BB/9', 'HR/9', 'BR/9'
#   return result

def predict(classifier, inputList):
  return classifier.predict(inputList)

# def printPredictionAccuracy(zippedList, featuresUsed):
#   correctCnt = 0
#   wrongCnt = 0
#   print '\n* Features used: '
#   for feature in featuresUsed:
#     print feature,
#   # print '\n'
#   for item in zippedList:
#     if item[0] == item[1]:
#       correctCnt += 1
#     else:
#       wrongCnt += 1
#   print "\n* Prediction Accuracy"
#   print " correct", correctCnt
#   print " wrong", wrongCnt
#   print " accuracy", 100 * correctCnt / (correctCnt + wrongCnt)

# START PARSING COMMAND LINE ARGUMENTS
target_letters = []
if len(sys.argv) >= 2:
  target_letters = sys.argv[1:]
else:
  target_letters = LETTERS
timed_log(target_letters)
# END PARSING COMMAND LINE ARGUMENTS

# START LOADING DATA AND CONCATINATING DATAFRAMES
# df = pd.DataFrame()
# for letter in target_letters:
#   file_name = FILE_NAME_PREFIX + letter + CSV
#   temp_df = pd.read_csv(file_name)
#   pieces = [df, temp_df]
#   df = pd.concat(pieces)
# timed_log('Dataset Size:', len(df))
# df.to_csv('whole' + CSV)
# END LOADING DATA AND CONCATINATING DATAFRAMES

# START SEPARATING MiLB DATA FROM MLB DATA
# df = pd.DataFrame.from_csv('whole' + CSV)
# minor_list = []
# major_list = []
# for index, row in df.iterrows():
#   minor_flag = is_minor_record(row)
#   if minor_flag:
#     minor_list.append(row)
#   else:
#     major_list.append(row)
# minor_df = pd.DataFrame.from_records(minor_list)
# major_df = pd.DataFrame.from_records(major_list)
# timed_log('MiLB Dataset Size:', len(minor_df))
# timed_log('MLB Dataset Size:', len(major_df))
# minor_df.to_csv('minor' + CSV)
# major_df.to_csv('major' + CSV)
# END SEPARATING MiLB DATA FROM MLB DATA

# START AGGREGATING MiLB DATA PER PLAYER
# minor_df = pd.DataFrame.from_csv('minor' + CSV)
# major_df = pd.DataFrame.from_csv('major' + CSV)
# minor_agg_list = []
# past_pid = None
# minor_agg_record_cnt = 0
# for letter in target_letters:
#   timed_log('processing', letter)
#   for index, row in minor_df.iterrows():
#     current_pid = row['PID']
#     if past_pid != None and past_pid == current_pid:
#       continue
#     else:
#       # WE HAVE A NEW PLAYER
#       player_minor_df = minor_df.query('PID == "' + current_pid + '"', engine='python')
#       if len(player_minor_df) == 0:
#         continue
#       player_major_df = major_df.query('PID == "' + current_pid + '"', engine='python')
#       player_minor_agg_record = aggregate(player_minor_df, MINOR_COPY, MINOR_PER_YEAR)
#       player_major_agg_record = aggregate(player_major_df, MAJOR_COPY, MAJOR_PER_YEAR)
#       player_minor_agg_record['WAR'] = player_major_agg_record['WAR']
#       minor_agg_list.append(player_minor_agg_record)
#       minor_agg_record_cnt += 1
#       past_pid = current_pid
# timed_log(minor_agg_record_cnt, len(minor_agg_list))
# minor_agg_df = pd.DataFrame.from_records(minor_agg_list)
# minor_agg_df.to_csv('minor_agg' + CSV)
# END AGGREGATING MiLB DATA PER PLAYER

# START VALIDATE & LABELING
# minor_agg_df = pd.DataFrame.from_csv('minor_agg' + CSV)
# minor_agg_df = pd.DataFrame(minor_agg_df, columns=DEFAULT_AGGR_RECORD.keys() + ['WARover8', 'WARover5', 'WARover2', 'WARover0'])
# minor_agg_df.fillna('', inplace=True)
# for index, row in minor_agg_df.iterrows():
#   for field in DEFAULT_AGGR_RECORD:
#     if row[field] == '':
#       row[field] = DEFAULT_AGGR_RECORD[field]
#   if row['WAR'] >= 8:
#     # MVP
#     row['WARover8'] = 1
#   else:
#     row['WARover8'] = 0
#   if row['WAR'] >= 5:
#     # ALL-STAR
#     row['WARover5'] = 1
#   else:
#     row['WARover5'] = 0
#   if row['WAR'] >= 2:
#     # STARTER
#     row['WARover2'] = 1
#   else:
#     row['WARover2'] = 0
#   if row['WAR'] >= 0:
#     # SUB
#     row['WARover0'] = 1
#   else:
#     row['WARover0'] = 0
# minor_agg_df.to_csv('minor_ready' + CSV, columns=DEFAULT_AGGR_RECORD.keys() + ['WARover8', 'WARover5', 'WARover2', 'WARover0'])
# timed_log('done')
# END VALIDATE & LABELING

# START TRAINING
minor_ready_df = pd.DataFrame.from_csv('minor_ready' + CSV)
minor_ready_df = shuffle_df(minor_ready_df)
train_size = int(math.floor(len(minor_ready_df) * 0.8))
test_size = len(minor_ready_df) - train_size
timed_log('Size of the dataset:', len(minor_ready_df))
timed_log('Size of the training set:', train_size)
timed_log('Size of the test set:', test_size)
train_df = minor_ready_df.iloc[:train_size]
test_df = minor_ready_df.iloc[train_size:]
train_df.to_csv('train_all_feat.csv')
test_df.to_csv('test_all_feat.csv')
train_input = []
train_predict_8 = train_df['WARover8'].tolist()
train_predict_5 = train_df['WARover5'].tolist()
train_predict_2 = train_df['WARover2'].tolist()
train_predict_0 = train_df['WARover0'].tolist()
for index, row in train_df.iterrows():
  del row['PID']
  del row['NAME']
  del row['WAR']
  del row['WARover8']
  del row['WARover5']
  del row['WARover2']
  del row['WARover0']
  train_input.append(row)
clf8 = getTrainedSVM(train_input, train_predict_8)
pickle.dump(clf8, open('clf8_all_feat.clf', 'wb'))
timed_log('8 training done')
clf5 = getTrainedSVM(train_input, train_predict_5)
pickle.dump(clf5, open('clf5_all_feat.clf', 'wb'))
timed_log('5 training done')
clf2 = getTrainedSVM(train_input, train_predict_2)
pickle.dump(clf2, open('clf2_all_feat.clf', 'wb'))
timed_log('2 training done')
clf0 = getTrainedSVM(train_input, train_predict_0)
pickle.dump(clf0, open('clf0_all_feat.clf', 'wb'))
timed_log('0 training done')
# END TRAINING

# START TESTING
test_df = pd.DataFrame.from_csv('test_all_feat.csv')
label_8 = test_df['WARover8'].tolist()
label_5 = test_df['WARover5'].tolist()
label_2 = test_df['WARover2'].tolist()
label_0 = test_df['WARover0'].tolist()
test_input = []
for index, row in test_df.iterrows():
  del row['PID']
  del row['NAME']
  del row['WAR']
  del row['WARover8']
  del row['WARover5']
  del row['WARover2']
  del row['WARover0']
  test_input.append(row)

clf8 = pickle.load(open('clf8_all_feat.clf', 'rb'))
pred_8_res = predict(clf8, test_input)
correct_cnt = 0
wrong_cnt = 0
for i in range(len(pred_8_res)):
  if label_8[i] == pred_8_res[i]:
    correct_cnt += 1
  else:
    wrong_cnt += 1
timed_log("correct", correct_cnt)
timed_log("wrong", wrong_cnt)
timed_log("accuracy", 100 * correct_cnt / (correct_cnt + wrong_cnt))

clf5 = pickle.load(open('clf5_all_feat.clf', 'rb'))
pred_5_res = predict(clf5, test_input)
correct_cnt = 0
wrong_cnt = 0
for i in range(len(pred_5_res)):
  if label_5[i] == pred_5_res[i]:
    correct_cnt += 1
  else:
    wrong_cnt += 1
timed_log("correct", correct_cnt)
timed_log("wrong", wrong_cnt)
timed_log("accuracy", 100 * correct_cnt / (correct_cnt + wrong_cnt))

clf2 = pickle.load(open('clf2_all_feat.clf', 'rb'))
pred_2_res = predict(clf2, test_input)
correct_cnt = 0
wrong_cnt = 0
for i in range(len(pred_2_res)):
  if label_2[i] == pred_2_res[i]:
    correct_cnt += 1
  else:
    wrong_cnt += 1
timed_log("correct", correct_cnt)
timed_log("wrong", wrong_cnt)
timed_log("accuracy", 100 * correct_cnt / (correct_cnt + wrong_cnt))

clf0 = pickle.load(open('clf0_all_feat.clf', 'rb'))
pred_0_res = predict(clf0, test_input)
correct_cnt = 0
wrong_cnt = 0
for i in range(len(pred_0_res)):
  if label_0[i] == pred_0_res[i]:
    correct_cnt += 1
  else:
    wrong_cnt += 1
timed_log("correct", correct_cnt)
timed_log("wrong", wrong_cnt)
timed_log("accuracy", 100 * correct_cnt / (correct_cnt + wrong_cnt))
# END TESTING

# testInput = combineInputs(testSetDf, featuresTest)
# predictRes = predict(classifier, testInput) # with default parameters
# table = []
# for i in range(len(predictRes)):
#   table.append([startId + i, predictRes[i]])
# results = pd.DataFrame(table, columns=['passenger_id', 'survived'])
# results.to_csv('output-default.csv')

# features = ['age', 'firstClass', 'secondClass', 'thirdClass', 'male', 'female']
# featuresTraining = list(trainingSetDf)
# featuresTraining = list(set(featuresTraining) & set(features))
# featuresTest = list(testSetDf)
# featuresTest = list(set(featuresTest) & set(features))

# trainingInput = combineInputs(trainingSetDf, featuresTraining)
# trainingPredicts = getColumnList(trainingSetDf, 'survived')
# classifier = getTrainedSVM(trainingInput, trainingPredicts)

# testInput = combineInputs(testSetDf, featuresTest)
# predictRes = predict(classifier, testInput) # with default parameters
# table = []
# for i in range(len(predictRes)):
#   table.append([startId + i, predictRes[i]])
# results = pd.DataFrame(table, columns=['passenger_id', 'survived'])
# results.to_csv('output-default.csv')



# features = ['age', 'firstClass', 'secondClass', 'thirdClass', 'male', 'female']
# featuresTraining = list(trainingSetDf)
# featuresTraining = list(set(featuresTraining) & set(features))
# featuresTest = list(testSetDf)
# featuresTest = list(set(featuresTest) & set(features))

# trainingInput = combineInputs(trainingSetDf, featuresTraining)
# trainingPredicts = getColumnList(trainingSetDf, 'survived')
# classifier = getTrainedSVM(trainingInput, trainingPredicts, C=0.3)

# testInput = combineInputs(testSetDf, featuresTest)
# predictRes = predict(classifier, testInput) # with default parameters
# table = []
# for i in range(len(predictRes)):
#   table.append([startId + i, predictRes[i]])
# results = pd.DataFrame(table, columns=['passenger_id', 'survived'])
# results.to_csv('output-with-lower-c.csv')




# features = ['age', 'firstClass', 'secondClass', 'thirdClass', 'male', 'female']
# featuresTraining = list(trainingSetDf)
# featuresTraining = list(set(featuresTraining) & set(features))
# featuresTest = list(testSetDf)
# featuresTest = list(set(featuresTest) & set(features))

# trainingInput = combineInputs(trainingSetDf, featuresTraining)
# trainingPredicts = getColumnList(trainingSetDf, 'survived')
# classifier = getTrainedSVM(trainingInput, trainingPredicts, C=3)

# testInput = combineInputs(testSetDf, featuresTest)
# predictRes = predict(classifier, testInput) # with default parameters
# table = []
# for i in range(len(predictRes)):
#   table.append([startId + i, predictRes[i]])
# results = pd.DataFrame(table, columns=['passenger_id', 'survived'])
# results.to_csv('output-with-higher-c.csv')




# features = ['age', 'firstClass', 'secondClass', 'thirdClass', 'male', 'female']
# featuresTraining = list(trainingSetDf)
# featuresTraining = list(set(featuresTraining) & set(features))
# featuresTest = list(testSetDf)
# featuresTest = list(set(featuresTest) & set(features))

# trainingInput = combineInputs(trainingSetDf, featuresTraining)
# trainingPredicts = getColumnList(trainingSetDf, 'survived')
# classifier = getTrainedSVM(trainingInput, trainingPredicts, gamma=8)

# testInput = combineInputs(testSetDf, featuresTest)
# predictRes = predict(classifier, testInput) # with default parameters
# table = []
# for i in range(len(predictRes)):
#   table.append([startId + i, predictRes[i]])
# results = pd.DataFrame(table, columns=['passenger_id', 'survived'])
# results.to_csv('output-higher-gamma.csv')




# features = ['age', 'firstClass', 'secondClass', 'thirdClass', 'male', 'female']
# featuresTraining = list(trainingSetDf)
# featuresTraining = list(set(featuresTraining) & set(features))
# featuresTest = list(testSetDf)
# featuresTest = list(set(featuresTest) & set(features))

# trainingInput = combineInputs(trainingSetDf, featuresTraining)
# trainingPredicts = getColumnList(trainingSetDf, 'survived')
# classifier = getTrainedSVM(trainingInput, trainingPredicts, gamma=2)

# testInput = combineInputs(testSetDf, featuresTest)
# predictRes = predict(classifier, testInput) # with default parameters
# table = []
# for i in range(len(predictRes)):
#   table.append([startId + i, predictRes[i]])
# results = pd.DataFrame(table, columns=['passenger_id', 'survived'])
# results.to_csv('output-med-gamma.csv')




# features = ['age', 'firstClass', 'secondClass', 'thirdClass', 'male', 'female']
# featuresTraining = list(trainingSetDf)
# featuresTraining = list(set(featuresTraining) & set(features))
# featuresTest = list(testSetDf)
# featuresTest = list(set(featuresTest) & set(features))

# trainingInput = combineInputs(trainingSetDf, featuresTraining)
# trainingPredicts = getColumnList(trainingSetDf, 'survived')
# classifier = getTrainedSVM(trainingInput, trainingPredicts, gamma=0.5)

# testInput = combineInputs(testSetDf, featuresTest)
# predictRes = predict(classifier, testInput) # with default parameters
# table = []
# for i in range(len(predictRes)):
#   table.append([startId + i, predictRes[i]])
# results = pd.DataFrame(table, columns=['passenger_id', 'survived'])
# results.to_csv('output-small-gamma.csv')





# features = ['age', 'firstClass', 'secondClass', 'thirdClass', 'male', 'female']
# featuresTraining = list(trainingSetDf)
# featuresTraining = list(set(featuresTraining) & set(features))
# featuresTest = list(testSetDf)
# featuresTest = list(set(featuresTest) & set(features))

# trainingInput = combineInputs(trainingSetDf, featuresTraining)
# trainingPredicts = getColumnList(trainingSetDf, 'survived')
# classifier = getTrainedSVM(trainingInput, trainingPredicts, gamma=0.25)

# testInput = combineInputs(testSetDf, featuresTest)
# predictRes = predict(classifier, testInput) # with default parameters
# table = []
# for i in range(len(predictRes)):
#   table.append([startId + i, predictRes[i]])
# results = pd.DataFrame(table, columns=['passenger_id', 'survived'])
# results.to_csv('output-smaller-gamma.csv')





# features = ['age', 'firstClass', 'secondClass', 'thirdClass', 'male', 'female']
# featuresTraining = list(trainingSetDf)
# featuresTraining = list(set(featuresTraining) & set(features))
# featuresTest = list(testSetDf)
# featuresTest = list(set(featuresTest) & set(features))

# trainingInput = combineInputs(trainingSetDf, featuresTraining)
# trainingPredicts = getColumnList(trainingSetDf, 'survived')
# classifier = getTrainedSVM(trainingInput, trainingPredicts, gamma=0.375)

# testInput = combineInputs(testSetDf, featuresTest)
# predictRes = predict(classifier, testInput) # with default parameters
# table = []
# for i in range(len(predictRes)):
#   table.append([startId + i, predictRes[i]])
# results = pd.DataFrame(table, columns=['passenger_id', 'survived'])
# results.to_csv('output-gamma-375.csv')




# features = ['age', 'firstClass', 'secondClass', 'thirdClass', 'male', 'female']
# featuresTraining = list(trainingSetDf)
# featuresTraining = list(set(featuresTraining) & set(features))
# featuresTest = list(testSetDf)
# featuresTest = list(set(featuresTest) & set(features))

# trainingInput = combineInputs(trainingSetDf, featuresTraining)
# trainingPredicts = getColumnList(trainingSetDf, 'survived')
# classifier = getTrainedSVM(trainingInput, trainingPredicts, gamma=0.125)

# testInput = combineInputs(testSetDf, featuresTest)
# predictRes = predict(classifier, testInput) # with default parameters
# table = []
# for i in range(len(predictRes)):
#   table.append([startId + i, predictRes[i]])
# results = pd.DataFrame(table, columns=['passenger_id', 'survived'])
# results.to_csv('output-gamma-125.csv')
