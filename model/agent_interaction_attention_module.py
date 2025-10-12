import torch.nn as nn
import torch
from timm.models.layers import trunc_normal_
from ptflops import get_model_complexity_info  
from fvcore.nn import FlopCountAnalysis, parameter_count

"""
The feature flow of the CFP branch interacts with the feature flow of the HFC branch, enhancing the perception of lesion details.
"""


class AgentInteractionAttention(nn.Module):
    r""" 
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 shift_size=0,
                 agent_num=49,
                 **kwargs):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.shift_size = shift_size

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, window_size[0], window_size[1]))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, window_size[0], window_size[1]))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0], 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1]))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

    def forward(self, x, x_2):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            x_2: high frequency component of x
        """
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads
        # print(f"head_dim:{head_dim}")
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        qkv_2 = self.qkv(x_2).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q_2, k_2, v_2 = qkv_2[0], qkv_2[1], qkv_2[2]
        # q, k, v: b, n, c
        agent_tokens = self.pool(q.reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)

        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        q_2 = q_2.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_2 = k_2.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v_2 = v_2.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        position_bias1 = nn.functional.interpolate(self.an_bias, size=self.window_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2

        agent_attn = self.softmax((agent_tokens * self.scale) @ k_2.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v_2

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2

        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x_ = q_attn @ agent_v


        x_ = x_.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x_ = x_ + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)
        x = x + x_

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, x_2

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'




# Package the model to support dual input.


class AgentInteractionAttentionWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        x, x_2 = inputs
        return self.module(x, x_2)

# Calculate the number of parameters and FLOPs
def calculate_params_and_flops_with_double_inputs(model, input_shapes):
    input_resolver = lambda: (
        torch.randn(input_shapes[0]),  # first input
        torch.randn(input_shapes[1])   # second input
    )

    macs, params = get_model_complexity_info(
        model,
        input_resolver,
        as_strings=True,  # The result is returned as a string.
        print_per_layer_stat=True,  # Print the statistics for each layer.
        verbose=True  # Use detailed mode
    )
    print(f"Total parameters: {params}")
    print(f"Total MACs: {macs}")


if __name__ == '__main__':
    # Initialization module
    dim = 384
    window_size = (14, 14)
    num_heads = 12
    agent_num = 9
    module = AgentInteractionAttention(dim=dim, window_size=window_size, num_heads=num_heads, agent_num=agent_num)
    # Packaging module
    wrapped_module = AgentInteractionAttentionWrapper(module)
    # input shape
    batch_size = 1
    num_patches = 196 
    input_dim = dim
    x = torch.randn((batch_size, num_patches, input_dim))  
    x_2 = torch.randn((batch_size, num_patches, input_dim)) 

    flops = FlopCountAnalysis(module, (x, x_2))  
    params = parameter_count(module)

    print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")  
    print(f"Total parameters: {sum(p.numel() for p in module.parameters()) / 1e6:.2f} M")

