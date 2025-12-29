import torch
from torch import nn
from torch.optim import AdamW
from networks.utils.mlp_builder import mlp_builder

class StaticObjectExtractor(nn.Module):
    def __init__(self, lr, wd, in_dim, enc_arch, out_dim, enc_act, temp = 0.1):
        super().__init__()
        self.enc = mlp_builder(in_dim, net_architecture = enc_arch, out_dim = out_dim, nonlinearity_name = enc_act)
        self.optimizer = AdamW(self.enc.parameters(), lr = lr, weight_decay = wd)
        self.temperature = temp

    def calculate_loss(self, obs, next_obs):
        # Batch x obj_qty x feat
        features = self.enc(obs)
        next_features = self.enc(next_obs)
        batch_size, obj_qty, feat_dim = features.shape
        feat1, feat2 = torch.unsqueeze(features, dim = 1), torch.unsqueeze(next_features, dim = 2)
        norm_feat1, norm_feat2 = nn.functional.normalize(feat1, dim = -1), nn.functional.normalize(feat2, dim = -1)
        cos_sim = torch.exp(torch.sum(norm_feat1 * norm_feat2, dim = -1) / self.temperature)
        draw_near = torch.diagonal(cos_sim, dim1 = -2, dim2 = -1)
        draw_apart = torch.sum(cos_sim, dim=-1) - draw_near
        contrastive_loss = torch.mean(torch.sum(-1 * torch.log(draw_near / (draw_apart + 1e-5)), dim = -1))
        return contrastive_loss, {'Similarity term': torch.mean(draw_near), 
                                  'Dissimilarity term': torch.mean(draw_apart)}

    def optimize_oe(self, obs, next_obs):
        loss, mets = self.calculate_loss(obs, next_obs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return mets
    
    def extract(self, obs):
        with torch.no_grad():
            return self.enc(obs)
        
class GTShapesObjectExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def optimize_oe(self, obs, next_obs):
        pass

    def extract(self, obs):
        return obs[..., :-2]

class IdentityExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def optimize_oe(self, obs, next_obs):
        pass

    def extract(self, obs):
        return obs
    
def get_object_extractor(type, lr, wd, in_dim, enc_arch, out_dim, enc_act, temp = 0.1):
    if type == 'classic':
        return StaticObjectExtractor(lr = lr, wd = wd, in_dim = in_dim, enc_arch = enc_arch,
                                     out_dim = out_dim, enc_act = enc_act, temp = temp)
    elif type == 'gt':
        return GTShapesObjectExtractor()
    
    elif type == 'Identity':
        return IndentationError()