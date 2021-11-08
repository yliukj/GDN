from utils import *
from scipy import sparse
import numpy as np
import networkx as nx
import pickle as pkl

def load_nci():
  with open("data/Ciao/adj.pkl", "rb") as f:
    adj = pkl.load(f)
  with open("data/Ciao/training_data.pkl", "rb") as f:
    train_data = pkl.load(f)
  with open("data/Ciao/testing_data.pkl", "rb") as f:
    test_data = pkl.load(f)
  return adj.astype(np.float32),train_data['User_list_train'],train_data['Product_list_train'],train_data['Rating_list_train'],test_data['User_list_test'],test_data['Product_list_test'],test_data['Rating_list_test']

    
