import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration, l2norm, torch_recurrent_gated_delta_rule
)

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

embed = lm.embed_tokens(torch.tensor([9707])).unsqueeze(0)
hidden = embed

normed = layer0.input_layernorm(hidden)
qkv_raw = la.in_proj_qkv(normed)  # [1,1,6144]

# Simulate conv1d for first token (state=zeros, no bias)
# transpose for conv: [1, 6144, 1]
qkv_t = qkv_raw.transpose(1, 2)
# Using torch conv1d with zero-padded state
conv_out = F.silu(la.conv1d(qkv_t)[:, :, :1])
conv_out = conv_out.transpose(1, 2)  # [1, 1, 6144]

print(f'conv_out first5: {conv_out[0,0,:5].tolist()}, norm={conv_out.norm().item():.6f}')

# Split QKV
key_dim = 2048
value_dim = 2048
q, k, v = torch.split(conv_out, [key_dim, key_dim, value_dim], dim=-1)
q = q.reshape(1, 1, 16, 128)
k = k.reshape(1, 1, 16, 128)
v = v.reshape(1, 1, 16, 128)

print(f'q first5: {q[0,0,0,:5].tolist()}, q norm={q.norm().item():.6f}')
print(f'k first5: {k[0,0,0,:5].tolist()}, k norm={k.norm().item():.6f}')
print(f'v first5: {v[0,0,0,:5].tolist()}, v norm={v.norm().item():.6f}')

z = la.in_proj_z(normed).reshape(1, 1, 16, 128)
a = la.in_proj_a(normed)
b = la.in_proj_b(normed)
beta = b.sigmoid()
g = -la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias)

print(f'g first5: {g[0,0,:5].tolist()}')
print(f'beta first5: {beta[0,0,:5].tolist()}')

# Run recurrence
# The function expects: q[B,T,H,D], g[B,T,H,1], beta[B,T,H,1]
# It does transpose(1,2) internally → q[B,H,T,D], g[B,H,T,1]
# For single token: T=1
g_4d = g.reshape(1, 1, 16, 1)
beta_4d = beta.reshape(1, 1, 16, 1)

# Manual single-step recurrence instead
q_norm = l2norm(q, dim=-1, eps=1e-6)  # [1,1,16,128]
k_norm = l2norm(k, dim=-1, eps=1e-6)
q_s = q_norm[0,0]  # [16,128]
k_s = k_norm[0,0]
v_s = v[0,0]  # [16,128]
beta_s = beta[0,0].sigmoid() if False else beta[0,0]  # already sigmoided
g_decay = g[0,0].exp()  # [16], decay in (0,1)
scale = 1.0 / (128 ** 0.5)

state = torch.zeros(16, 128, 128)
# state *= decay (state is zero, so no-op)
# kv_mem[h,j] = sum_i state[h,i,j] * k_norm[h,i]
kv_mem = torch.einsum('hij,hi->hj', state, k_s)
delta = (v_s - kv_mem) * beta_s.unsqueeze(-1)
state = state + torch.einsum('hi,hj->hij', k_s, delta)
out_step = torch.einsum('hij,hi->hj', state, q_s * scale)
print(f'recurrent out first5: {out_step[0,:5].tolist()}, norm={out_step.norm().item():.6f}')
out = out_step.unsqueeze(0)  # [1,16,128]
print(f'recurrent out (reshaped) first5: {out[0,:5].tolist()}, norm={out.norm().item():.6f}')

# Norm
out_flat = out.reshape(-1, 128)  # [16, 128]
z_flat = z.reshape(-1, 128)  # [16, 128]
normed_out = la.norm(out_flat, z_flat)
print(f'normed_gated first5: {normed_out[0,:5].tolist()}, norm={normed_out.norm().item():.6f}')

normed_out_3d = normed_out.reshape(1, 1, -1)
proj_out = la.out_proj(normed_out_3d)
print(f'proj_out first5: {proj_out[0,0,:5].tolist()}, norm={proj_out.norm().item():.6f}')

# Final = hidden + proj_out
final = hidden + proj_out
print(f'after residual first5: {final[0,0,:5].tolist()}, norm={final.norm().item():.6f}')

# FFN
ffn_normed = layer0.post_attention_layernorm(final)
gate_out = layer0.mlp.gate_proj(ffn_normed)
up_out = layer0.mlp.up_proj(ffn_normed)
act = F.silu(gate_out) * up_out
down_out = layer0.mlp.down_proj(act)
layer_out = final + down_out
print(f'layer_0 output first5: {layer_out[0,0,:5].tolist()}, norm={layer_out.norm().item():.6f}')
