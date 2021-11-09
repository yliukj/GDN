import numpy as np
from one_fold import fold_cv
import tensorflow as tf



temp = []
ccfold = 1
for i in range(ccfold,2):
  tf.reset_default_graph()
  seed = 123
  np.random.seed(seed)
  tf.set_random_seed(seed)
  ttemp = fold_cv(cv_index = i)
  temp.append(ttemp)
  print(i,ttemp)
print("the RMSE of ciao is ","{:.5f}".format(sum(temp)/ccfold)) 
