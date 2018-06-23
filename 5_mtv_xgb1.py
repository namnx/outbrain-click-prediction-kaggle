import gc
import numpy as np
import feather
import xgboost as xgb

print('+ Loading trained models...')
model_fold0 = xgb.Booster({'nthread': 4})
model_fold0.load_model('tmp/xgb_model_0.model')
model_fold1 = xgb.Booster({'nthread': 4})
model_fold1.load_model('tmp/xgb_model_1.model')


print('+ Loading test data...')
df_test = feather.read_dataframe('tmp/mtv_df_test.feather')
features = sorted(set(df_test.columns) - {'display_id', 'clicked'})

X_test = df_test[features].values
del df_test
gc.collect()

dtest = xgb.DMatrix(X_test, feature_names=features)
del X_test
gc.collect()


print('+ Predicting using test data...')
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
