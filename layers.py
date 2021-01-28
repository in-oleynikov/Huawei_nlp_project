import torch
import torch.nn as nn
import torch.nn.functional as F


class AverageEmbedding(nn.Module):
    def __init__(self):#, maxlen, emb_dim):
        super().__init__()
    
    def forward(self, emb_x, dim=1):#, emb, neg):
        '''input_embedding: (batch_size, maxlen, embedding_dim)
           output_embedding: (batch_size, 1, embedding_dim)'''
#         mask = torch.sum(emb_x!=0, 1)
#         average_pos = 
        return torch.sum(emb_x, dim, keepdim=True)/torch.sum(emb_x!=0, dim, keepdim=True)


class AttentionLayer(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        
        self.W = torch.empty(emb_dim, emb_dim, requires_grad=True)#, dtype=torch.float32)
        self.W = torch.nn.init.xavier_uniform_(self.W)
        self.W = torch.nn.Parameter(self.W)
        
    def forward(self, emb_x, average_sent_emb):
        d_x = torch.matmul(torch.matmul(emb_x, self.W), average_sent_emb.permute(0, 2, 1)) #(batch_size, maxlen, 1)
        attn_weights = F.softmax(d_x, dim=1) #(batch_size, maxlen, 1)
        
        return attn_weights

class WeightedEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, attention_weights, emb_x):
        """attention_weights.shape: batch_size, maxlen, 1
           emb_x.shape: batch_size, maxlen, 200"""
        return torch.sum(attention_weights * emb_x, 1, keepdim=True)


class WeightedAspects(nn.Module):
    def __init__(self, aspect_matrix):
        super().__init__()
        self.aspect_matrix = aspect_matrix
#         self.aspect_matrix = self.aspect_matrix.transpose(1,0)
        self.aspect_matrix = torch.nn.Parameter(self.aspect_matrix)
        
    def forward(self, aspect_weights):
        '''aspect_weights.shape: batch_size, 1, n_aspects'''
        return torch.matmul(aspect_weights, self.aspect_matrix)
#         return self.aspect_matrix * aspect_weights
