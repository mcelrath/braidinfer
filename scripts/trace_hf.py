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

activations = {}

def make_hook(name):
    def hook(mod, inp, out):
        if isinstance(out, tuple):
            activations[name] = out[0].detach()
        else:
            activations[name] = out.detach()
    return hook

lm = model.model.language_model
lm.embed_tokens.register_forward_hook(make_hook('embed'))
for i in range(24):
    lm.layers[i].register_forward_hook(make_hook(f'layer_{i}'))
lm.norm.register_forward_hook(make_hook('final_norm'))

input_ids = torch.tensor([[9707]])
with torch.no_grad():
    out = model(input_ids=input_ids)

for name in ['embed'] + [f'layer_{i}' for i in range(24)] + ['final_norm']:
    t = activations[name]
    if t.dim() == 3:
        t = t[0, 0]
    print(f'{name}: first5={t[:5].tolist()}, norm={t.norm().item():.6f}')
