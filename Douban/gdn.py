import tensorflow as tf
from layers import *


class GDN(object):
  def build_graph(self, placeholders, n=3000, d=3000, hidden_d=256,u=64,reuse=False):
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
      
      self.hidden = GraphConvolutionSparse(input_dim=self.d,
                                      output_dim=hidden_d,
                                      #adj=gcn_adj,
                                      adj=self.adj,
                                      features_nonzero=self.features_nonzero,
                                      act=tf.nn.relu,
                                      dropout=self.dropout,
                                      logging=False)
      hidden = self.hidden(self.input_F)
      inverse_H = hidden
      de_hidden = GraphConvolution(input_dim=hidden_d,
                                           output_dim=hidden_d,
                                           #adj=gcn_adj_inverse,
                                           adj=self.adj_inverse,
                                           act=tf.nn.relu,
                                           #act=lambda x:x,
                                           logging=False)(inverse_H)
      oooo = de_hidden
      self.reconstruct = GraphConvolution(input_dim=hidden_d,
                                           output_dim=self.d,
                                           adj=self.adj,
                                           #adj=gcn_adj,
                                           act=lambda x:x,
                                           logging=False)
      self.reconstruct_X = self.reconstruct(oooo)



  def trainable_vars(self):
    return [var for var in
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='SelfAttentiveGraph')]
