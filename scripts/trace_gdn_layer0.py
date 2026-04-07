import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

config = AutoConfig.from_pretrained('Qwen/Qwen3.5-0.8B')
config.vocab_size = config.text_config.vocab_size
config.hidden_size = config.text_config.hidden_size
config.pad_token_id = getattr(config.text_config, 'pad_token_id', None)

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    'Qwen/Qwen3.5-0.8B', config=config, torch_dtype=torch.float32
)
model.eval()

lm = model.model.language_model
layer0 = lm.layers[0]
la = layer0.linear_attn

embed = lm.embed_tokens(torch.tensor([9707])).unsqueeze(0)  # [1,1,1024]
hidden = embed

# RMSNorm
normed = layer0.input_layernorm(hidden)  # [1,1,1024]
print(f'normed: first5={normed[0,0,:5].tolist()}, norm={normed.norm().item():.6f}')

# in_proj_qkv
qkv = la.in_proj_qkv(normed)  # [1,1,6144]
print(f'qkv (pre-conv): first5={qkv[0,0,:5].tolist()}, norm={qkv.norm().item():.6f}')

# conv1d
# For the HF implementation, conv1d is applied to qkv
# Check how HF does the conv1d
print(f'conv1d weight shape: {la.conv1d.weight.shape}')
print(f'conv1d weight first: {la.conv1d.weight[0,0,:].tolist()}')

# The HF GDN forward pass - let me trace what the actual forward does
# Look at the class
print(f'layer0 type: {type(layer0)}')
print(f'linear_attn type: {type(la)}')

# For a single token decode, conv state is zeros, so conv1d output = input * weight[-1]
# (only the last weight column applies since state is all zeros and input is position 0)
# After SiLU activation
w_last = la.conv1d.weight[:, 0, -1]  # [6144]
conv_out_manual = qkv[0, 0] * w_last
silu_out = conv_out_manual * torch.sigmoid(conv_out_manual)
print(f'manual conv+silu: first5={silu_out[:5].tolist()}, norm={silu_out.norm().item():.6f}')

# Project a, b, z
a_proj = la.in_proj_a(normed)  # [1,1,16]
b_proj = la.in_proj_b(normed)  # [1,1,16]
z_proj = la.in_proj_z(normed)  # [1,1,2048]
print(f'a_proj: {a_proj[0,0,:5].tolist()}')
print(f'b_proj: {b_proj[0,0,:5].tolist()}')
print(f'z_proj first5: {z_proj[0,0,:5].tolist()}, norm={z_proj.norm().item():.6f}')

# Gate: g = -exp(A_log) * softplus(a + dt_bias)
A = la.A_log  # [16]
dt = la.dt_bias  # [16]
print(f'A_log: {A[:5].tolist()}')
print(f'dt_bias: {dt[:5].tolist()}')
gate = -torch.exp(A) * torch.nn.functional.softplus(a_proj[0,0] + dt)
print(f'gate: {gate[:5].tolist()}')
