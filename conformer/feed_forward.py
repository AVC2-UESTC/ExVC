# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch import Tensor

from .activation import Swish
from .modules import Linear
from torch.nn import functional as F


class Expert(nn.Module):
    def __init__(self, encoder_dim, expansion_factor=4,dropout_p=0.1):
        super(Expert, self).__init__()
        self.sequential = nn.Sequential(
             nn.LayerNorm(encoder_dim),
             Linear(encoder_dim,encoder_dim, bias=True),
             Swish(),
             nn.Dropout(p=dropout_p))

    def forward(self, inputs):
        return  (self.sequential(inputs.transpose(1,2))).transpose(1,2)
       
class GatingNetwork(nn.Module):
    def __init__(self, input_dim,num_experts,dropout_rate):
        super(GatingNetwork, self).__init__()                    
        self.layer1 = nn.Conv1d(input_dim, num_experts,1)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.layer1(x)
        x=self.leaky_relu(x)
        x = self.dropout1(x)
        x= x.transpose(1,2)  # transpose b,e,t,->b,t,e 
        return torch.softmax(x,dim=2) 
        
        
    
class MoELayer(nn.Module):
    """
    Defining the Mixture of Experts Layer class. 
    ExVC is based on the dense-MoE mechanism, which we found to perform better than the top-k approach. However, it can also be trained using 
    the top-k mechanism by setting the number of topk_indices to be different from the number of experts (provided you use a single GPU). In ExVC, the number of experts and 
    the number of topk_indices are both set to 4.  

    Acknowledgment: ruvnet/MoE.py  https://gist.github.com/ruvnet/0928768dd1e4af8816e31dde0a0205d5
         
    """
    
    def __init__(self, input_dim,output_dim, num_experts=4):
        super(MoELayer, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(num_experts)])
        self.gate = GatingNetwork(input_dim,num_experts,dropout_rate=0.1)

    def forward(self, x, num_experts_per_tok=4):

        # transpose b,t,d->b,d,t 
        x=x.transpose(1,2) 
        gate_scores = self.gate(x)  # gating score is of the form b,t,e; here e is the number of experts. 
        topk_gating_scores, topk_indices = gate_scores.topk(num_experts_per_tok, dim=2, sorted=False)
        
        # Create a mask to zero out the contributions of non-topk experts,
        # In exVC,this mask is useless, since we need to use all the experts' outputs. It will always be 1s 
        mask = torch.zeros_like(gate_scores).scatter_(2, topk_indices, 1)      
        
        # Mask is used to retain the only topk gating scores
        gate_scores = gate_scores * mask
        
        # Normalize the gating scores to sum to 1 across the selected top experts
        gate_scores = F.normalize(gate_scores, p=1, dim=2)


        # the output of the each expert are of the form , b,d,t
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) #b,e,d,t  the stack of the outputs of all experts
        expert_outputs = torch.einsum('bedt->bted', expert_outputs)
        output = torch.einsum('bte,bted->btd', gate_scores, expert_outputs)   # multiplying each expert with its weighting score. 
        return output # b,t,d


# add MoELayer to FeedForwardModule_expert class 
class FeedForwardModule_expert(nn.Module):
    """
    Conformer Feed Forward Module follow pre-norm residual units and apply layer normalization within the residual unit
    and on the input before the first linear layer. This module also apply Swish activation and dropout, which helps
    regularizing the network.

    Args:
        encoder_dim (int): Dimension of conformer encoder
        expansion_factor (int): Expansion factor of feed forward module.
        dropout_p (float): Ratio of dropout

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor contains input sequences

    Outputs: outputs
        - **outputs** (batch, time, dim): Tensor produces by feed forward module.
    """
    def __init__(
            self,
            encoder_dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule_expert, self).__init__()
        self.moe_layer = MoELayer(encoder_dim,encoder_dim,num_experts=4)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.moe_layer(inputs)


class FeedForwardModule(nn.Module):
    """
    Conformer Feed Forward Module follow pre-norm residual units and apply layer normalization within the residual unit
    and on the input before the first linear layer. This module also apply Swish activation and dropout, which helps
    regularizing the network.

    Args:
        encoder_dim (int): Dimension of conformer encoder
        expansion_factor (int): Expansion factor of feed forward module.
        dropout_p (float): Ratio of dropout

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor contains input sequences

    Outputs: outputs
        - **outputs** (batch, time, dim): Tensor produces by feed forward module.
    """
    def __init__(
            self,
            encoder_dim: int = 192,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)
