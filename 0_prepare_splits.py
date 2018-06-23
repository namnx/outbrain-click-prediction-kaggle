
import pandas as pd
import numpy as np
import feather


# prepare train split
df_train = pd.read_csv("data/clicks_train.csv.zip",
                       dtype={'display_id': np.uint32, 'ad_id': np.uint32, 'clicked': np.uint8})

display_ids = df_train.display_id.unique()
np.random.seed(1)
np.random.shuffle(display_ids)
val_size = int(len(display_ids) * 0.5)
val_display_ids = set(display_ids[:val_size])

df_train['fold'] = 0
df_train.loc[df_train['display_id'].isin(val_display_ids), 'fold'] = 1
df_train.fold = df_train.fold.astype('uint8')
feather.write_dataframe(df_train, 'tmp/clicks_train_50_50.feather')


# prepare test data
df_test = pd.read_csv("data/clicks_test.csv.zip", dtype={'display_id': np.uint32, 'ad_id': np.uint32})
feather.write_dataframe(df_test, 'tmp/clicks_test.feather')
