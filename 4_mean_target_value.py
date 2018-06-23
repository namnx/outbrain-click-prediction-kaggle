# run with cat categorical_features.txt | parallel --jobs 6 python 4_mean_target_value.py {}
# coding: utf-8

import sys
from time import time
import pandas as pd
import numpy as np
import feather

column = sys.argv[1]
C = 12
print('processing column %s...' % column)

df_all = feather.read_dataframe('tmp/clicks_train_50_50.feather')
df_test = feather.read_dataframe('tmp/clicks_test.feather')

train_col = pd.read_csv('tmp/categorical/train/' + column + '.txt', header=None, dtype='str')
df_all[column] = train_col[0]

test_col = pd.read_csv('tmp/categorical/test/' + column + '.txt', header=None, dtype='str')
df_test[column] = test_col[0]

df_train_0 = df_all[df_all.fold == 0].reset_index(drop=1)
df_train_1 = df_all[df_all.fold == 1].reset_index(drop=1)
del df_train_0['fold'], df_train_1['fold'], df_all['fold']


# fold 0 train
print('\ntraining on fold 0, predicting on 1')
t0 = time()

m0 = (df_train_0.clicked == 1).mean()
cnt_clicked_0 = df_train_0[df_train_0.clicked == 1][column].value_counts()
cnt_all_0 = df_train_0[column].value_counts()

probs_1 = (cnt_clicked_0 + C * m0) / (cnt_all_0 + C)
probs_1 = probs_1[df_train_1[column]].reset_index(drop=1)
probs_1.fillna(m0, inplace=True)
#df_train_1['prob'] = probs_1
del cnt_clicked_0, cnt_all_0

print('took %0.3fs' % (time() - t0))


# fold 1 train
print('\ntraining on fold 1, predicting on 0')
t0 = time()

m1 = (df_train_1.clicked == 1).mean()
cnt_clicked_1 = df_train_1[df_train_1.clicked == 1][column].value_counts()
cnt_all_1 = df_train_1[column].value_counts()

probs_0 = (cnt_clicked_1 + C * m1) / (cnt_all_1 + C)
probs_0 = probs_0[df_train_0[column]].reset_index(drop=1)
probs_0.fillna(m1, inplace=True)
#df_train_0['prob'] = probs_0
del cnt_clicked_1, cnt_all_1

print('took %0.3fs' % (time() - t0))


# full train
print('\ntraining on all data, predicting on test')
t0 = time()

m = (df_all.clicked == 1).mean()
cnt_clicked = df_all[df_all.clicked == 1][column].value_counts()
cnt_all = df_all[column].value_counts()

probs = (cnt_clicked + C * m) / (cnt_all + C)
probs = probs[df_test[column]].reset_index(drop=1)
probs.fillna(m, inplace=True)
#df_test['prob'] = probs
del cnt_clicked, cnt_all, df_all

print('took %0.3fs' % (time() - t0))


# saving the results
np.save('features/mtv/' + column + '_pred_0.npy', probs_0.values)
np.save('features/mtv/' + column + '_pred_1.npy', probs_1.values)
np.save('features/mtv/' + column + '_pred_test.npy', probs.values)
del probs_0, probs_1, probs
del df_train_0['ad_id'], df_train_0['clicked']
del df_train_1['ad_id'], df_train_1['clicked']
del df_test['ad_id']


f = column
# creating the rank features
print('\ncreating the ranking features for df_train_0...')
t0 = time()
df_train_0['%s_rank' % f] = df_train_0.groupby('display_id')[f].rank(method='max', ascending=False)
df_train_0['%s_rank' % f] = df_train_0['%s_rank' % f].astype('uint8')
np.save('features/mtv/' + column + '_pred_rank_0.npy', df_train_0['%s_rank' % f].values)
del df_train_0
print('took %0.3fs' % (time() - t0))

print('\ncreating the ranking features for df_train_1...')
t0 = time()
df_train_1['%s_rank' % f] = df_train_1.groupby('display_id')[f].rank(method='max', ascending=False)
df_train_1['%s_rank' % f] = df_train_1['%s_rank' % f].astype('uint8')
np.save('features/mtv/' + column + '_pred_rank_1.npy', df_train_1['%s_rank' % f].values)
del df_train_1
print('took %0.3fs' % (time() - t0))

print('\ncreating the ranking features for df_test...')
t0 = time()
df_test['%s_rank' % f] = df_test.groupby('display_id')[f].rank(method='max', ascending=False)
df_test['%s_rank' % f] = df_test['%s_rank' % f].astype('uint8')
np.save('features/mtv/' + column + '_pred_rank_test.npy', df_test['%s_rank' % f].values)
del df_test
print('took %0.3fs' % (time() - t0))
