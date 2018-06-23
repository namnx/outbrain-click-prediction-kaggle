import gc
import numpy as np
import xgboost as xgb
import feather


print('+ Loading mtv_df_train_1...')
df_train_1 = feather.read_dataframe('tmp/mtv_df_train_1.feather')
features = sorted(set(df_train_1.columns) - {'display_id', 'clicked'})

X_1 = df_train_1[features].values
y_1 = df_train_1.clicked.values
del df_train_1
gc.collect()

dfold1 = xgb.DMatrix(X_1, y_1, feature_names=features)
del X_1, y_1
gc.collect()


print('+ Loading mtv_df_train_0...')
df_train_0 = feather.read_dataframe('tmp/mtv_df_train_0.feather')

y_0 = df_train_0.clicked.values
X_0 = df_train_0[features].values
del df_train_0
gc.collect()

dfold0 = xgb.DMatrix(X_0, y_0, feature_names=features)
del X_0, y_0
gc.collect()



# training a model
n_estimators = 100
xgb_pars = {
    'eta': 0.2,
    'gamma': 0.5,
    'max_depth': 6,
    'min_child_weight': 1,
    'max_delta_step': 0,
    'subsample': 1,
    'colsample_bytree': 0.5,
    'colsample_bylevel': 0.5,
    'lambda': 1,
    'alpha': 0,
    'tree_method': 'approx',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 20,
    'seed': 42,
    'silent': 1
}


print('+ Training model on fold 0...')
watchlist = [(dfold0, 'train'), (dfold1, 'val')]
model_fold1 = xgb.train(xgb_pars, dfold0, num_boost_round=n_estimators,
                        verbose_eval=1, evals=watchlist)
model_fold1.save_model('tmp/xgb_model_1.model')


print('+ Predicting on trained model (fold 0)...')
pred1 = model_fold1.predict(dfold1)
np.save('predictions/xgb_mtv_pred1.npy', pred1)
del pred1
gc.collect()


print('+ Saving the training leaves on fold 0...')
leaves1 = model_fold1.predict(dfold1, pred_leaf=True).astype('uint8')
np.save('tmp/xgb_model_1_leaves.npy', leaves1)
del leaves1
gc.collect()




print('+ Training model on fold 1...')
watchlist = [(dfold1, 'train'), (dfold0, 'val')]
model_fold0 = xgb.train(xgb_pars, dfold1, num_boost_round=n_estimators,
                        verbose_eval=1, evals=watchlist)
model_fold0.save_model('tmp/xgb_model_0.model')


print('+ Predicting on trained model (fold 1)...')
pred0 = model_fold0.predict(dfold0)
np.save('predictions/xgb_mtv_pred0.npy', pred0)
del pred0
gc.collect()


print('+ Saving the training leaves on fold 1...')
leaves0 = model_fold0.predict(dfold0, pred_leaf=True).astype('uint8')
np.save('tmp/xgb_model_0_leaves.npy', leaves0)
del leaves0
gc.collect()



# making prediction for test and getting the leaves
print('+ Making prediction for test and getting the leaves...')
'''
Traceback (most recent call last):
  File "5_mtv_xgb.py", line 110, in <module>
    dtest = xgb.DMatrix(X_test, feature_names=features)
  File "/usr/local/lib/python2.7/dist-packages/xgboost/core.py", line 278, in __init__
    self._init_from_npy2d(data, missing, nthread)
  File "/usr/local/lib/python2.7/dist-packages/xgboost/core.py", line 346, in _init_from_npy2d
    data = np.array(mat.reshape(mat.size), copy=False, dtype=np.float32)
MemoryError
Exception AttributeError: "'DMatrix' object has no attribute 'handle'" in <bound method DMatrix.__del__ of <xgboost.core.DMatrix object at 0x7fc9e1fbd150>> ignored
'''
df_test = feather.read_dataframe('tmp/mtv_df_test.feather')
X_test = df_test[features].values
del df_test
gc.collect()

dtest = xgb.DMatrix(X_test, feature_names=features)
del X_test
gc.collect()

pred0_test = model_fold0.predict(dtest)
pred1_test = model_fold1.predict(dtest)
pred_test = (pred0_test + pred1_test) / 2

np.save('predictions/xgb_mtv_pred_test.npy', pred_test)
del pred0_test, pred1_test, pred_test
gc.collect()


# predicting leaves for test
print('+ Predicting leaves for test...')
leaves0_test = model_fold0.predict(dtest, pred_leaf=True).astype('uint8')
np.save('tmp/xgb_model_0_test_leaves.npy', leaves0_test)
del leaves0_test
gc.collect()


leaves1_test = model_fold1.predict(dtest, pred_leaf=True).astype('uint8')
np.save('tmp/xgb_model_1_test_leaves.npy', leaves1_test)
del leaves1_test
gc.collect()
