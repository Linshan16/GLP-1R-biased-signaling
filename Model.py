import math
import torch
from torch import nn
from torch.nn import LayerNorm
from esm.multihead_attention import MultiheadAttention


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class TransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        add_bias_kv=True,
        use_esm1b_layer_norm=False,
        use_rotary_embeddings: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        self._init_submodules(add_bias_kv, use_esm1b_layer_norm)

    def _init_submodules(self, add_bias_kv, use_esm1b_layer_norm):

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn


class RefinementLayers(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers: int = 33, embed_dim: int = 1280,  attention_heads: int = 20):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.output_dim = output_dim
        self._init_submodules()

    def _init_submodules(self):
        self.resize_input = nn.Linear(self.input_dim, self.embed_dim)
        self.layers = nn.ModuleList([TransformerLayer(self.embed_dim, self.embed_dim, self.attention_heads, add_bias_kv=False, use_esm1b_layer_norm=True, use_rotary_embeddings=True,) for _ in range(self.num_layers)])
        self.norm_after = LayerNorm(self.embed_dim)
        self.output = nn.Linear(self.embed_dim, self.output_dim)

    def forward(self, input_tensor, repr_layers=[], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True
        x = self.resize_input(input_tensor)
        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x
        if need_head_weights:
            attn_weights = []

        x = x.transpose(0, 1)
        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(x, need_head_weights=need_head_weights)
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                attn_weights.append(attn.transpose(1, 0))
        x = self.norm_after(x)
        x = x.transpose(0, 1)

        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        result = {"representations": hidden_representations}
        if need_head_weights:
            attentions = torch.stack(attn_weights, 1)
            result["attentions"] = attentions

        x = x.mean(dim=1)
        x = self.output(x)
        result["output"] = x
        return result


class DynamicFC(nn.Module):
    def __init__(self, input_dim, output_dim, layer_dim_list: list = [256], layer_norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.layer_dim_list = layer_dim_list
        self.layer_norm = layer_norm
        self.output_dim = output_dim
        self._init_submodules()

    def _init_submodules(self):
        self.layer_dim_list = [self.input_dim] + self.layer_dim_list
        self.layers = nn.ModuleList([])
        for idx, layer_dim in enumerate(self.layer_dim_list):
            if idx+1 < len(self.layer_dim_list):
                self.layers.append(nn.Linear(layer_dim, self.layer_dim_list[idx+1]))
                self.layers.append(nn.ReLU())
                if self.layer_norm:
                    self.layers.append(LayerNorm(self.layer_dim_list[idx+1]))
        self.output = nn.Linear(self.layer_dim_list[-1], self.output_dim)

    def forward(self, x):
        result = {}
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x)
        x = x.mean(dim=1)
        x = self.output(x)
        result["output"] = x
        return result
