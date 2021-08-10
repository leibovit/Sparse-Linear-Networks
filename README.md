# Sparse Linear Networks with a Fixed Butterfly Structure: Theory and Practice

Code to accompany the paper **Sparse Linear Networks with a Fixed Butterfly Structure: Theory and Practice**


A 16 × 16 butterfly network represented as product of 4 sparse matrices            |  A 16 × 16 butterfly network represented as a 4-layered graph
:-------------------------:|:-------------------------:
![](./Images/mats.png)  |  ![](./Images/BF.png)

## Requirements

- [x] python>=3.6
- [x] pytorch>=1.8
- [x] numpy
- [x] scipy

## Example Usage

Let's run the proposed replacement for a Dense Linear Layer over a simple one hidden layer Neural Network. All you need to do is to use `BF` instead of `nn.Linear` in any of your models.

```python
from butterfly import Butterfly

# Definition of BF
def BF(input_dim,output_dim,k1,k2):
    n1,n2 = input_dim, output_dim
    first_gadget = Butterfly(in_size=n1, out_size=k1)
    second_gadget = nn.Linear(k1,k2)
    third_gadget = Butterfly(in_size=k2, out_size= n2)
    
    return nn.Sequential(first_gadget,second_gadget,third_gadget)

# Proposed replacement
from torch import nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer (linear transformation
        k1 = 10
        k2 = 8
        self.hidden = BF(784, 256,k1,k2) # nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        k1 = 8
        k2 = 3
        self.output = BF(256, 10,k1,k2) #nn.Linear(256, 10)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x
```
Done! The network now has Butterfly layers.

## Citations

```BibTeX
@inproceedings{
Sparse Linear Networks,
title={Sparse Linear Networks with a Fixed Butterfly Structure: Theory and Practice},
author={Omer Leibovitch and Vineet Nair and Nir Ailon},
year={2021},
booktitle={the 37th Conference on Uncertainty in Artificial Intelligence (UAI 2021)}
}
```
