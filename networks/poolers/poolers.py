import torch
from torch import nn
from math import sqrt

class FetcherPooler(nn.Module):
    def __init__(self, obs_length, skill_length = None):
        super().__init__()
        self.expect_skill = False
        self.outp_dim = obs_length
        if skill_length is not None:
            self.expect_skill = True
            self.outp_dim += skill_length

    def forward(self, seq, skill = None, obj_idx = None):
        # Expecting seq to be of shape [Batch, Seq_len, obs_dim]
        outp = seq[torch.arange(0, seq.shape[0], 1), obj_idx]
        if self.expect_skill:
            outp = torch.cat([outp, skill], dim = -1)
        return outp
    
class TransformerPooler(nn.Module):
    def __init__(self, obs_length, skill_length, nhead = 4, dim_feedforward = 256, num_layers = 2):
        super().__init__()
        self.projector = nn.Linear(obs_length + skill_length, dim_feedforward)
        self.q = nn.Linear(obs_length, dim_feedforward) 
        self.k = nn.Linear(dim_feedforward, dim_feedforward)
        self.v = nn.Linear(dim_feedforward, dim_feedforward)
        _transformer_pooler = nn.TransformerEncoderLayer(dim_feedforward, 
                                                         nhead = nhead, dim_feedforward = dim_feedforward, 
                                                         batch_first = True, norm_first = False)
        self.transformer_pooler = nn.TransformerEncoder(_transformer_pooler, num_layers = num_layers)
        """
        self.readout_token = nn.Parameter(torch.randn(size = (1, 1, dim_feedforward)) * 0.05)
        self.skill_token = nn.Parameter(torch.randn((1, 1, dim_feedforward)) * 1/3)
        """
        self.obs_dim = obs_length
        self.dim_feedforward = dim_feedforward
        self.outp_dim = dim_feedforward + skill_length
        
    def forward(self, seq, skill = None, obj_idx = None):
        # Expecting seq to be of shape [Batch, Seq_len, obs_dim]
        batch_len, seq_len, obs_dim = seq.shape
        # Add readout token
        aligned_skill = torch.unsqueeze(skill, dim = 1).repeat((1, seq_len, 1))
        skill_seq = torch.cat((seq, aligned_skill), axis = -1)
        processed_seq = self.projector(skill_seq)
        # Add mark (skill token) so pooler knows to which object skill is applied
        # Returns processed output token
        """
        seq = torch.cat((self.readout_token.expand(batch_len, -1, -1), seq), dim = 1)
        """
        transformed_seq = self.transformer_pooler(processed_seq)
        query = torch.unsqueeze(self.q(seq[torch.arange(batch_len), obj_idx, :]), dim = 1)
        keys, values = self.k(transformed_seq), self.v(transformed_seq)
        attention = torch.softmax(torch.sum(query * keys, axis=-1, keepdim=True)/sqrt(self.dim_feedforward), dim = -2)
        output = torch.sum(attention * values, axis=-2)
        return torch.cat([output, skill], dim=-1)
    
class ConcatPooler(nn.Module):
    def __init__(self, obs_length, obj_qty, skill_length = None):
        super().__init__()
        self.expect_skill = False
        self.outp_dim = obs_length * obj_qty
        if skill_length is not None:
            self.outp_dim += skill_length
            self.expect_skill = True

    def forward(self, seq, skill = None, obj_idx = None):
        batch_len, seq_len, obs_dim = seq.shape
        embed = torch.cat([seq[:, i] for i in range(seq_len)], axis = -1)
        if skill is not None:
            embed = torch.cat([embed, skill], axis = -1)
        return embed
    
def get_pooler_network(name, obs_length, skill_length, pooler_config, obj_qty = None):
    if name == 'Transformer':
        return TransformerPooler(obs_length = obs_length, skill_length = skill_length, **pooler_config)
    elif name == 'Fetcher':
        return FetcherPooler(obs_length = obs_length)
    elif name == 'Concat':
        return ConcatPooler(obs_length = obs_length, skill_length = skill_length, obj_qty = obj_qty)