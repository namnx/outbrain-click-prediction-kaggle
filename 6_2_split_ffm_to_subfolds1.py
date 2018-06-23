import numpy as np
from tqdm import tqdm


print('+ split fold 1 into subfolds...')

subfold_1 = np.load('tmp/fold_1_split.npy')
f_0 = open('ffm/ffm_xgb_1_0.txt', 'w')
f_1 = open('ffm/ffm_xgb_1_1.txt', 'w')

with open('ffm/ffm_xgb_1.txt', 'r') as f_in:
    for fold, line in tqdm(zip(subfold_1, f_in)):
        if fold == 0:
            f_0.write(line)
        else:
            f_1.write(line)

f_0.close()
f_1.close()
