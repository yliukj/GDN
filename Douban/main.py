import numpy as np
from one_fold import fold_cv
import tensorflow as tf



tf.reset_default_graph()
temp = []
for i in range(1):
  result = fold_cv()
  temp.append(result)
print("the RMSE of douban is ","{:.5f}".format(sum(temp)/len(temp))) 
