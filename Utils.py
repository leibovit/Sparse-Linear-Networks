import torch
import torch.nn as nn

import numpy as np
from butterfly import Butterfly


# In[22]:


def calc_k_function(n1,n2):

    k1 = max(n1 // 5 , (int(np.log2(n1)) ** 2) + 1)
    k2 = max(n2 // 5 , (int(np.log2(n2)) ** 2) + 1)
    return k1, k2

# Definition of BF

def BF(input_dim,output_dim):

    n1,n2 = input_dim, output_dim
    k1,k2 = calc_k_function(n1,n2)
    first_gadget = Butterfly(in_size=n1, out_size=n1, bias=False, complex=False,
                              tied_weight=False, increasing_stride=True, ortho_init=True)
    second_gadget = nn.Linear(n1,k1,bias=False)
    third_gadget = Butterfly(in_size=k1, out_size = k1, bias=False, complex=False,
                              tied_weight=False, increasing_stride=True, ortho_init=True)
    
    return nn.Sequential(first_gadget,second_gadget,third_gadget)

