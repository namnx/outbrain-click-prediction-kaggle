''' adapted from https://www.kaggle.com/its7171/leakage-solution
'''
import sys
import csv
import gc
import numpy as np
import pandas as pd
csv.field_size_limit(sys.maxsize)


# compute ad ctr based on training set
print('+ Loading training set and computing ad ctr...')
df_train = pd.read_csv('data/clicks_train.csv.zip')
ad_id_ctr_df = df_train.groupby('ad_id', as_index=False).agg({'display_id': 'count', 'clicked': 'sum'})
ad_id_ctr_df.columns = ['ad_id', 'views', 'clicks']
ad_id_ctr_df['ctr'] = ad_id_ctr_df['clicks'] / ad_id_ctr_df['views']
avg_ctr = ad_id_ctr_df['ctr'].mean()

try: del df_train
except: gc.collect()


# loading leaked docs
print('+ Loading leaked docs...')
with open("tmp/leaked_docs.csv") as f:
    reader = csv.DictReader(f)
    leaks = {}

    for row in reader:
        doc_id = int(row['document_id'])
        uuids = row['uuids'].split(' ')
        leaks[doc_id] = set(uuids)


# compute results based on ad ctr and leaked docs
print('+ Loading test set and using leaked docs...')
df_test = pd.read_csv('data/clicks_test.csv.zip')
df_promoted_content = pd.read_csv('data/promoted_content.csv.zip',
                                  usecols=('ad_id', 'document_id'),
                                  dtype={'ad_id': np.int, 'uuid': np.str, 'document_id': np.str})
df_events = pd.read_csv('data/events.csv.zip',
                        usecols=('display_id', 'uuid'),
                        dtype={'display_id': np.int, 'uuid': np.str})

df_test = pd.merge(df_test, df_promoted_content, on='ad_id', how='left')
df_test = pd.merge(df_test, df_events, on='display_id', how='left')
df_test['is_leak'] = df_test.apply(lambda row: row['document_id'] in leaks
                                            and row['uuid'] in leaks[row['document_id']], 
                              axis = 1)
print(df_test['is_leak'].value_counts())


print('+ Sorting and grouping data...')
del df_test['document_id']
del df_test['uuid']
df_test = pd.merge(df_test, ad_id_ctr_df, how='left', on='ad_id')
df_test.at[df_test['views'].isnull(), 'ctr'] = avg_ctr
df_test.at[df_test['clicks'] == 0, 'ctr'] = -1
df_test.at[df_test['views'] < 5, 'ctr'] = avg_ctr
df_test.at[df_test['is_leak'], 'ctr'] = 1

df_test = df_test.sort_values(by=['display_id', 'ctr'], ascending=[True, False])
df_test = df_test.groupby('display_id', as_index=False).agg({'ad_id': lambda x: ' '.join([str(i) for i in x])})

print('+ Exporting result...')
df_test.to_csv('1_Leaks.csv.gz', index=False, compression='gzip', chunksize=1024)
