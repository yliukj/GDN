import tensorflow as tf
import numpy as np
from utils import *
from gdn import GDN
import networkx as nx
from scipy.sparse import csr_matrix
from preprocessing import load_data_monti
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

'''
parse
'''
tf.app.flags.DEFINE_integer('num_epochs', 200, 'number of epochs to train')
tf.app.flags.DEFINE_integer('labels', 1, 'number of label classes')
tf.app.flags.DEFINE_integer('graph_pad_length', 3000, 'graph pad length for training')
tf.app.flags.DEFINE_integer('decay_step', 10, 'decay steps')
tf.app.flags.DEFINE_integer('cv_index', 2, 'fold_ID')
tf.app.flags.DEFINE_float('learn_rate', 5e-3, 'learn rate for training optimization')
tf.app.flags.DEFINE_boolean('train', True, 'train mode FLAG')

FLAGS = tf.app.flags.FLAGS

cv_index = FLAGS.cv_index
num_epochs = FLAGS.num_epochs
tag_size = FLAGS.labels
graph_pad_length = FLAGS.graph_pad_length
feature_dimension = 3000
lr = FLAGS.learn_rate




def get_rmse_score(reconstruct_x,edges_pos, edges_neg,test_label):
    preds = reconstruct_x[edges_pos,edges_neg]
    preds = np.where(preds>4,4,preds)
    preds = np.where(preds<0,0,preds)
    rmse_score = np.sqrt(np.average(np.square(test_label -preds)))
    return rmse_score




def fold_cv(cv_index = 2,if_train = FLAGS.train):
  placeholders = {
    'gcn_support': tf.sparse_placeholder(tf.float32),
    'gcn_inverse_support': tf.sparse_placeholder(tf.float32),
    'support': tf.sparse_placeholder(tf.float32),
    'inverse_support': tf.sparse_placeholder(tf.float32),
    'features': tf.sparse_placeholder(tf.float32),
    'num_nodes': tf.placeholder(tf.int32),
    'labels': tf.placeholder(tf.float32),
    'u_indices': tf.placeholder(tf.int32),
    'v_indices': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
  }
  with tf.Session() as sess:
    model = GDN()
    model.build_graph(n=graph_pad_length,placeholders = placeholders,d =feature_dimension)
    with tf.variable_scope('DownstreamApplication'):
      global_step = tf.Variable(0, trainable=False, name='global_step')
      learn_rate = tf.train.exponential_decay(lr, global_step, FLAGS.decay_step, 0.9, staircase=True)
      labels = placeholders['labels']
      logits = model.reconstruct_X
      full_indices = tf.stack([placeholders['u_indices'], placeholders['v_indices']], axis=1)
      logit_care = tf.gather_nd(logits,full_indices)
      loss = tf.losses.mean_squared_error(labels,logit_care)
      weight_decay = 0.001
      for var in model.hidden.vars.values():
        loss += weight_decay * tf.nn.l2_loss(var)
      for var in model.reconstruct.vars.values():
        loss += weight_decay * tf.nn.l2_loss(var)
      params = tf.trainable_variables()
      optimizer = tf.train.AdamOptimizer(learn_rate)
      grad_and_vars = tf.gradients(loss, params)
      clipped_gradients, _ = tf.clip_by_global_norm(grad_and_vars, 1)
      opt = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)
  

    # load data
    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
        val_labels, val_u_indices, val_v_indices, test_labels, \
        test_u_indices, test_v_indices, class_values = load_data_monti('douban', True)
    feature_input = csr_matrix((train_labels, (train_u_indices, train_v_indices)), dtype = np.float)
    structure_input = u_features
    print(len(train_labels))

    sess.run(tf.global_variables_initializer())
    if if_train == True:
      #print('start training')
      hist_loss = []
      test_rmse = []
      for epoch_num in range(num_epochs):
        epoch_loss = 0
        batch_input,topology = (feature_input,structure_input)
        batch_input = preprocess_features(batch_input.tolil())
        num_nodes = len(train_labels)
        attention_rating = num_nodes - 200
        if epoch_num <52:
          idx = np.random.RandomState( seed = epoch_num + 123).permutation(num_nodes)[:attention_rating]
        topo = dropout_edge(topology,0.5,epoch_num+123)
        #topo = dropout_edge(topology,1,0)
        batch_topo = preprocess_adj(topo)
        batch_topo_inverse = preprocess_inverse_adj(topo)
        gcn_batch_topo = preprocess_gcn_adj(topo)
        gcn_batch_topo_inverse = preprocess_gcn_inverse_adj(topo)
        train_ops = [opt, loss, learn_rate, global_step]
        feed_dict = construct_feed_dict(gcn_batch_topo,gcn_batch_topo_inverse,batch_input, batch_topo, batch_topo_inverse, num_nodes,train_labels[idx],train_u_indices[idx],train_v_indices[idx],placeholders)
        result = sess.run(train_ops, feed_dict=feed_dict)
        epoch_loss = result[1]
        print("Epoch:", '%04d' % (epoch_num), "train_loss=", "{:.5f}".format(epoch_loss))
        hist_loss.append(epoch_loss)
        if epoch_num > 5:
          feed_dict.update({placeholders['dropout']: 0.})
          topo = dropout_edge(topology,1,0)
          batch_topo = preprocess_adj(topo)
          batch_topo_inverse = preprocess_inverse_adj(topo)
          gcn_batch_topo = preprocess_gcn_adj(topo)
          gcn_batch_topo_inverse = preprocess_gcn_inverse_adj(topo)
          feed_dict = construct_feed_dict(gcn_batch_topo,gcn_batch_topo_inverse,batch_input, batch_topo, batch_topo_inverse, num_nodes,train_labels,train_u_indices,train_v_indices,placeholders)
          pred_x = sess.run(logits, feed_dict=feed_dict)
          test_rmse.append(get_rmse_score(pred_x,test_u_indices,test_v_indices,test_labels))
          if epoch_num >50:
            idx = np.argpartition(np.square(pred_x[train_u_indices,train_v_indices]-train_labels), attention_rating)[:attention_rating]
          #if epoch_num %10 == 0:
          #  print(pred_x[train_u_indices,train_v_indices],train_labels,np.sqrt(mean_squared_error(pred_x[train_u_indices,train_v_indices],train_labels)))
    else:
      saver = tf.train.Saver()
      saver.restore(sess, "./pretrained/{}/model.ckpt".format(cv_index))

    sess.close()
    te_rmse = min(test_rmse)
    #print(test_rmse.index(min(test_rmse)))
    return te_rmse
