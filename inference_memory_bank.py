import math
import torch
from model.positional_encodings import PositionalEncodingPermute3D
import torch.nn.functional as F

def softmax(x):
    x_exp = x.exp_()
    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    return x_exp

def softmax_w_top(x, top):
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = F.softmax(values,dim=1)

    #x_exp = values.exp_()
    #x_exp /= torch.sum(x_exp, dim=1, keepdim=True)

    # The types should be the same already
    # some people report an error here so an additional guard is added
    x.zero_().scatter_(1, indices, x_exp.type(x.dtype)) # B * THW * HW

    return x

def softmax_w_kmn(x, h, w, sigma):
    _, memery_ele_n, _ = x.shape

    max_query_index = torch.argmax(x,dim=2)

    # convert max index to 2d position in query image
    max_query_y = (max_query_index//w).reshape(1,-1,1).repeat(1,1,h*w)
    max_query_x = (max_query_index%w).reshape(1,-1,1).repeat(1,1,h*w)

    # meshgrid
    gridy,gridx = torch.meshgrid(torch.Tensor(range(h)),torch.Tensor(range(w)))
    gridy = gridy.reshape(1,1,-1).repeat(1,memery_ele_n,1).cuda()
    gridx = gridx.reshape(1,1,-1).repeat(1,memery_ele_n,1).cuda()

    g = torch.exp(-((gridy - max_query_y) ** 2 + (gridx - max_query_x) ** 2)/(2 * sigma ** 2))

    x_exp = x.exp_() * g
    values, indices = torch.topk(x_exp, k=20, dim=1)

    values /= torch.sum(values, dim=1, keepdim=True)

    x.zero_().scatter_(1, indices, values.type(x.dtype))


    return x

class MemoryBank:
    def __init__(self, k, top_k=20, sigma = 7, memory_type = 'topk'):
        self.memory_type = memory_type
        self.top_k = top_k
        self.sigma = sigma

        self.CK = None
        self.CV = None

        self.mem_k = None
        self.mem_v = None

        self.num_objects = k
        self.PE = PositionalEncodingPermute3D(128 * 3)

    def _global_matching(self, mk, qk, mpe, qpe, h, w):
        # NE means number of elements -- typically T*H*W
        B, CK, NE = mk.shape

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk

        # add cosine similarity of space time positional encoding
        pab = (mpe/torch.norm(mpe,dim=1,keepdim=True)).transpose(1,2) @ (qpe/torch.norm(qpe,dim=1,keepdim=True))
        #affinity = (2*ab-a_sq) * pab / math.sqrt(CK)   # B, NE, HW

        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, NE, HW


        assert self.memory_type in ['normal', 'topk', 'kmn']

        if self.memory_type == 'normal':
            affinity = softmax(affinity)  # B, NE, HW
        elif self.memory_type == 'topk':
            affinity = softmax_w_top(affinity, top=self.top_k)  # B, NE, HW
        elif self.memory_type == 'kmn':
            affinity = softmax_w_kmn(affinity, h, w, sigma=self.sigma)  # B, NE, HW

        return affinity

    def _readout(self, affinity, mv):
        return torch.bmm(mv, affinity)

    def match_memory(self, qk):
        k = self.num_objects
        _, _, h, w = qk.shape

        qk = qk.flatten(start_dim=2)
        
        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 2)
            mv = torch.cat([self.mem_v, self.temp_v], 2)
        else:
            mk = self.mem_k
            mv = self.mem_v

        mt = mk.shape[-1]//(h*w)
        pe_tensor = torch.zeros((1,128 * 3, mt + 1, h, w))
        pe = self.PE(pe_tensor).cuda()
        mpe = pe[:,:,:-1,:,:].flatten(start_dim=2)
        qpe = pe[:,:,-1,:,:].flatten(start_dim=2)

        affinity = self._global_matching(mk, qk, mpe, qpe, h, w)

        # One affinity for all
        readout_mem = self._readout(affinity.expand(k,-1,-1), mv)

        return readout_mem.view(k, self.CV, h, w)

    def add_memory(self, key, value, is_temp=False):
        # Temp is for "last frame"
        # Not always used
        # But can always be flushed
        self.temp_k = None
        self.temp_v = None
        key = key.flatten(start_dim=2)
        value = value.flatten(start_dim=2)

        if self.mem_k is None:
            # First frame, just shove it in
            self.mem_k = key
            self.mem_v = value
            self.CK = key.shape[1]
            self.CV = value.shape[1]
        else:
            if is_temp:
                self.temp_k = key
                self.temp_v = value
            else:
                self.mem_k = torch.cat([self.mem_k, key], 2)
                self.mem_v = torch.cat([self.mem_v, value], 2)
