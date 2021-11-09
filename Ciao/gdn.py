import tensorflow as tf
from layers import *


class GDN(object):
  def build_graph(self, placeholders, n=3000, d=3000, hidden_d=256,u=128,reuse=False):
    with tf.variable_scope('SelfAttentiveGraph', reuse=reuse):
      self.n = n
      self.d = d
      self.u = u
      self.dropout = placeholders['dropout']
      gcn_adj = placeholders['gcn_support']
      gcn_adj_inverse = placeholders['gcn_inverse_support']
      self.adj = placeholders['support']
      self.adj_inverse = placeholders['inverse_support']
      self.input_F = placeholders['features']
      self.features_nonzero = placeholders['num_nodes']
      self.placeholders = placeholders
      
      hidden = GraphConvolutionSparse(input_dim=self.d,
                                      output_dim=hidden_d,
                                      #adj=gcn_adj,
                                      adj=self.adj,
                                      features_nonzero=self.features_nonzero,
                                      act=tf.nn.relu,
                                      dropout=self.dropout,
                                      logging=False)(self.input_F)

      self.H = GraphConvolution(input_dim=hidden_d,
                                           output_dim=self.u,
                                           #adj=gcn_adj,
                                           adj=self.adj,
                                           #act=lambda x: x,
                                           act=tf.nn.selu,
                                           logging=False)(hidden)
      hhhh = tf.concat([self.H,hidden],axis=1)
      inverse_H = hhhh
      de_hidden = GraphConvolution(input_dim=3*self.u,
                                           output_dim=hidden_d,
                                           #adj=gcn_adj_inverse,
                                           adj=self.adj_inverse,
                                           act=tf.nn.relu,
                                           logging=False)(inverse_H)
      self.reconstruct_X = GraphConvolution(input_dim=hidden_d,
                                           output_dim=self.d,
                                           adj=self.adj,
                                           #adj=gcn_adj,
                                           act=lambda x:x,
                                           logging=False)(de_hidden)



  def trainable_vars(self):
    return [var for var in
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='SelfAttentiveGraph')]
