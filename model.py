import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from layers import AverageEmbedding, AttentionLayer, WeightedEmbeddings, WeightedAspects

from sklearn.cluster import KMeans
import numpy as np

TORCH_DEVICE = 'cpu'#'cuda' #

def get_aspect_matrix(emb_matrix, n_clusters):
        km = KMeans(n_clusters=n_clusters)
        km.fit(emb_matrix)
        clusters = km.cluster_centers_

        # L2 normalization
        norm_aspect_matrix = clusters / np.linalg.norm(clusters, axis=-1, keepdims=True)
        return norm_aspect_matrix.astype(np.float64)

class PosNegDataset(Dataset):
    def __init__(self, data, target=None, transform=None, neg_size=2, seed=0):
        self.data = data #torch.from_numpy(data).long()
        self.transform = transform
        self._single_sample_len = len(self.data[0])
        self._neg_size = neg_size

#         should I fix seed?
#         np.random.seed(seed)
        self._idxs = np.array(list(range(len(self))))

    def __getitem__(self, index):
        pos = self.data[index]
        
        # TODO: avoid same object in positive and negative samples
        neg_samples = np.random.choice(self._idxs, size=self._neg_size, replace=False)
        neg = torch.reshape(
            self.data[neg_samples],
            (self._neg_size, self._single_sample_len)
        )
        return pos, neg
    
    def __len__(self):
        return len(self.data)

    
    
class Net(nn.Module):
    def __init__(self, vocab_size=300, emb_dim = 300, maxlen=10, n_aspects=10,
                 pretrained_embeddings=None, aspect_matrix=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.maxlen = maxlen
        self.n_aspects = n_aspects
        self.aspect_matrix = torch.from_numpy(aspect_matrix,).to(TORCH_DEVICE).requires_grad_(requires_grad=True)
        
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True, 
                                                      padding_idx=0).to(TORCH_DEVICE).requires_grad_(requires_grad=True)
                                                      #(voc_size, emb_dim)
        self.average_emb = AverageEmbedding() #(maxlen, emb_dim)
        self.attention = AttentionLayer(emb_dim)
        self.weighted_emb = WeightedEmbeddings()
        self.linear =  nn.Linear(emb_dim, n_aspects)
        self.weighted_aspects = WeightedAspects(self.aspect_matrix)

        
    def forward(self, x_pos, x_neg): 
        '''x_pos.shape: batch_size, maxlen
           x_neg.shape: batch_size, neg_size, maxlen'''
        #input: (batch_size, maxlen)
        emb_pos = self.embedding(x_pos) #batch_size, maxlen, emb_dim
        emb_neg = self.embedding(x_neg) #batch_size, neglen, maxlen, emb_dim
        
        y_s = self.average_emb(emb_pos) #batch_size, 1, emb_dim
        attn_weights = self.attention(emb_pos, y_s) #batch_size, maxlen, 1
        z_s = self.weighted_emb(attn_weights, emb_pos) #batch_size, 1, emb_dim
        
        z_n = self.average_emb(emb_neg, dim=2) #batch_size, neglen, 1, emb_dim
        
        
        p_t = self.linear(z_s) #batch_size, 1, n_aspects
        p_t = F.softmax(p_t, dim=2, dtype=torch.float64)
        r_s = self.weighted_aspects(p_t).type_as(z_s) #batch_size, 1, emb_dim

        
        return r_s, z_s, z_n, p_t, attn_weights
    
    
class MaxMarginLoss(nn.Module):
    def __init__(self, lamb=10):
        super().__init__()
        self.lamb = lamb
    
    def forward(self, r_s, z_s, z_n, aspect_matrix):
        '''r_s: (batch_size, embedding_dim, 1)
           z_s: (batch_size, embedding_dim, 1)
           z_n: (batsh_size, embedding_dim, n_negative_samples, 1)'''
        
        n_neg_samples = z_n.shape[1]
        z_n.squeeze_(2)
        
        z_s = F.normalize(z_s)
        r_s = F.normalize(r_s)
        z_n = F.normalize(z_n)
        
        r_s_exp = r_s.expand(-1, n_neg_samples, -1)
        
        for_max = 1. - torch.sum(r_s*z_s, dim=2).expand(-1, n_neg_samples) + torch.sum(r_s*z_n, dim=2)
#         one = torch.ones_like(for_max)
#         for_max = one - for_max
        zero_or = torch.zeros(2, len(for_max.flatten()))
        zero_or[0] = for_max.flatten()
        maxi = torch.max(zero_or, 0).values.reshape_as(for_max)
        loss = torch.sum(torch.sum(maxi, dim=1), dim=0)
        
        regularization = torch.matmul(aspect_matrix, aspect_matrix.permute(1,0)) - \
        torch.eye(aspect_matrix.shape[0], aspect_matrix.shape[0], dtype=torch.float64, device=TORCH_DEVICE, requires_grad=True)
        regularization = regularization.norm()
        
        loss = loss + self.lamb*regularization
                
        return loss
