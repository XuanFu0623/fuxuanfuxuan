# copied from CompVis/latent-diffusion
# https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/attention.py
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from .diffusion_utils import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)





class VisualReliabilityGate(nn.Module):
    """
    视觉可靠性门控机制
    提取视觉特征的可靠性信息并生成门控值
    支持直接提供质量特征或从视觉特征中估计
    """
    def __init__(self, feature_dim=64, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, visual_features, quality_features=None):
        """
        输入视觉特征，输出门控值
        visual_features: [batch_size, seq_len, feature_dim]
        quality_features: [batch_size, seq_len, feature_dim] 可选，直接提供的质量特征
        return: [batch_size, seq_len, 1]
        """
        # 如果提供了质量特征，则直接使用
        if quality_features is not None:
            features = quality_features
        else:
            # 否则从视觉特征中估计
            features = visual_features
            
        # 计算门控值 (batch_size, seq_len, 1)
        gate_values = self.mlp(features)
        return gate_values

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=4, dim_head=4, dropout=0., use_visual_gating=True): #8,64
        super().__init__()
        inner_dim = dim_head * heads
        self.context_dim = context_dim
        context_dim = default(context_dim, query_dim) #4096?why, 64 32]
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.use_visual_gating = use_visual_gating

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_k = nn.Linear(64, inner_dim, bias=False) 
        self.to_v = nn.Linear(64, inner_dim, bias=False)
        
        # 添加视觉可靠性门控机制
        if use_visual_gating:
            self.visual_gate = VisualReliabilityGate(feature_dim=64)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, visual_quality=None, quality_features=None):
        h = self.heads

        q = self.to_q(x)
        
        # if self.context_dim==256:
        #context = default(context, x)
        # elif self.context_dim==64:
        context = default(context, q)
        
        # 应用视觉可靠性门控
        if self.use_visual_gating and context is not None:
            # 如果提供了视觉质量特征，则直接使用
            if visual_quality is None:
                # 如果提供了质量特征，则使用质量特征计算门控值
                visual_quality = self.visual_gate(context, quality_features)
            
            # 应用门控值到视觉特征 (Key和Value)
            # 确保门控值维度正确 [batch, seq_len, 1]
            if visual_quality.dim() == 2:
                gate_values = visual_quality.unsqueeze(-1)
            else:
                gate_values = visual_quality
                
            # 按元素相乘，门控视觉特征
            context = context * gate_values
        
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        #print('att')
        x = self.attn1(self.norm1(x)) + x
        #print('att2')
        x = self.attn2(self.norm2(x), context=context) + x # is self-attn if context is none
        #print('ff')
        x = self.ff(self.norm3(x)) + x
        return x

#### Latent-diffusion 모델에서 Cross-attention용으로 쓰였던 모듈...
#### Q: from z_T (=noisy sample) / K, V: from condition embedding (여기서 context 인자로 넣어줌)
class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, res=None):
        super().__init__()
        self.in_channels = in_channels
        if res is None:
            res = in_channels
        inner_dim = n_heads * d_head # 4*4=16
        self.norm = Normalize(in_channels)
        #self.proj_in = nn.Conv2d(in_channels,
        #                         inner_dim,
        #                         kernel_size=1,
        #                         stride=1,
         #                        padding=0)
        #self.proj_in2 = nn.Conv2d(inner_dim,
        #                         1,
        #                         kernel_size=1,
        #                         stride=1,
        #                         padding=0)
        
        #self.transformer_blocks = nn.ModuleList(
        #    [BasicTransformerBlock(res, n_heads, d_head, dropout=dropout, context_dim=context_dim) # context_dim=32 원래는 inner_dim*res
        #        for d in range(depth)]
        #)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(res, n_heads, d_head, dropout=dropout, context_dim=context_dim) # context_dim=32 원래는 inner_dim*res
                for d in range(depth)]
        )

        #self.proj_out2 = zero_module(nn.Conv2d(1,
        #                                      inner_dim,
        #                                      kernel_size=1,
        #                                      stride=1,
        #                                      padding=0))
        #self.proj_out = zero_module(nn.Conv2d(inner_dim,
        #                                      in_channels,
        #                                      kernel_size=1,
        #                                      stride=1,
        #                                      padding=0))
        
        if res==64:
            self.proj1 = nn.Linear(256,64)
            self.proj2 = nn.Linear(64,256)
        elif res==128:
            self.proj1 = nn.Linear(256,128)
            self.proj2 = nn.Linear(128,256)
        else:
            self.proj1 = nn.Linear(256,32)
            self.proj2 = nn.Linear(32,256)
            
    def forward(self, x, context=None):
        x_in = x # for residual connection
        x = self.norm(x) # [8, 256, 64, 64]
        # x = self.proj_in(x) # [8, 64, 64, 64]
        # x = self.proj_in2(x) #[8, 1, 64, 64]
        # b, c, f, t = x.shape
        # #x = rearrange(x, 'b c h w -> b (h w) c') # [8, 4096, 64] # h,w=freq,time -> 우리도 spec을 이미지 취급하면 괜찮을것같긴 한데..... 근데 
        # x = rearrange(x, 'b c f t -> b t (c f)')
        # for block in self.transformer_blocks:
        #     x = block(x, context=context)
        # x = rearrange(x, 'b t (c f) -> b c f t', f=f, c=c) # [8, 64, 64] -> [8, 1, 64, 64]
        # x = self.proj_out2(x) #[8, 64, 64, 64]
        # x = self.proj_out(x) # [8, 256, 64, 64]
        
        x = x.mean(dim=2).transpose(1,2) #[4,256,64] dim2 ->1
        x = self.proj1(x)
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = self.proj2(x).transpose(1,2) #[4,64,256]
        #x = x.transpose(1,2)
        x = x.unsqueeze(2) # 2 ->1로
        return x + x_in



