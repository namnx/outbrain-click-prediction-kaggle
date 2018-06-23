import gc
import numpy as np
import feather
from sklearn.ensemble import ExtraTreesClassifier


print('+ Loading df_train_1...')
df_train_1 = feather.read_dataframe('tmp/mtv_df_train_1.feather')
del df_train_1['display_id']
train_1_shape_0 = df_train_1.shape[0]


print('+ Preparing data for full model...')
print('  - clicked')
df_train_0 = feather.read_dataframe('tmp/mtv_df_train_1.feather', columns=['clicked'])
train_0_shape_0 = df_train_0.shape[0]
y = np.concatenate((df_train_1.clicked.values, df_train_0.clicked.values))
del df_train_1['clicked'], df_train_0
gc.collect()


features = sorted(set(df_train_1.columns))
X = np.empty((train_1_shape_0 + train_1_shape_0, len(features)))
for i, f in enumerate(features):
    print('  - [%d/%d] %s' % (i+1, len(features), f))
    X[:train_1_shape_0, i] = df_train_1[f].values
    del df_train_1[f]
    gc.collect()

    df_train_0 = feather.read_dataframe('tmp/mtv_df_train_1.feather', columns=[f])
    X[train_1_shape_0:, i] = df_train_0[f].values
    del df_train_0
    gc.collect()

del df_train_1
gc.collect()


print('+ Training on full dataset...')
et_params = dict(
    criterion='entropy',
    max_depth=40,
    min_samples_split=6,
    min_samples_leaf=6,
    max_features=6,
    bootstrap=False,
    n_jobs=-1,
    random_state=1
)

et_full = ExtraTreesClassifier(warm_start=True, **et_params)
et_full.n_estimators = 100
et_full.fit(X, y)
del X, y
gc.collect()



print('+ Making predictions for test...')

df_test = feather.read_dataframe('tmp/mtv_df_test.feather', columns=features)
X_test = df_test[features].values
del df_test

pred_test = et_full.predict_proba(X_test)[:, 1].astype('float32')
np.save('predictions/et_pred_test.npy', pred_test)
