# import sonnet as snt
# from tensor2tensor.layers import common_attention
# from tensor2tensor.layers import common_layers
# import tensorflow.compat.v1 as tf
# from tensorflow.python.framework import function
# import tensorflow_probability as tfp

import numpy as np
import torch.nn as nn
import utils.layer_utils as layer_utils
import torch
import utils.data_utils_torch as data_utils
import math
import os

### Basic modeuls ###
# Transformer Encoder #
# Transformer Decoder #

# transformer encoder for 
class TransformerEncoder(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 fc_size=1024,
                 num_heads=4,
                 layer_norm=True,
                 num_layers=8,
                 dropout_rate=0.2,
                 re_zero=True,
                 memory_efficient=False,
                 ):
        super(TransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.fc_size = fc_size
        self.num_heads = num_heads
        self.num_heads = 1
        self.layer_norm = layer_norm
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.re_zero = re_zero
        self.memory_efficient = memory_efficient

        ### Attention layer and related modules ###
        self.attention_layers = nn.ModuleList()
        if self.layer_norm:
            self.layer_norm_layers = nn.ModuleList()
        if self.re_zero:
            self.re_zero_vars = nn.ParameterList()
        if self.dropout_rate: # dropout rate
            self.dropout_layers = nn.ModuleList()
        for i in range(self.num_layers):
            cur_atten_layer = nn.MultiheadAttention(
                self.hidden_size, self.num_heads, dropout=0.0, bias=True, kdim=self.hidden_size, vdim=self.hidden_size, batch_first=True)
            self.attention_layers.append(cur_atten_layer)
            if self.layer_norm:
                cur_layer_norm = nn.LayerNorm(self.hidden_size)
                self.layer_norm_layers.append(cur_layer_norm)
            if self.re_zero:
                cur_re_zero_var = torch.nn.Parameter(torch.zeros(size=(1,), dtype=torch.float32, requires_grad=True), requires_grad=True)
                self.re_zero_vars.append(cur_re_zero_var)
            if self.dropout_rate:
                cur_dropout_layer = nn.Dropout(p=self.dropout_rate)
                self.dropout_layers.append(cur_dropout_layer)

        ### Attention layer and related modules ###
        self.fc_layers = nn.ModuleList()
        if self.layer_norm:
            self.fc_layer_norm_layers = nn.ModuleList()
        if self.re_zero:
            self.fc_re_zero_vars = nn.ParameterList()
        if self.dropout_rate:
            self.fc_dropout_layers = nn.ModuleList() # dropout layers
        for i in range(self.num_layers):
            cur_fc_layer = nn.Linear(in_features=self.hidden_size, out_features=self.fc_size, bias=True)
            cur_fc_layer_2 = nn.Linear(in_features=self.fc_size, out_features=self.hidden_size, bias=True)
            self.fc_layers.append(nn.Sequential(*[cur_fc_layer, cur_fc_layer_2]))
            if self.layer_norm: # layer norm
                cur_layer_norm = nn.LayerNorm(self.hidden_size)
                self.fc_layer_norm_layers.append(cur_layer_norm)
            if self.re_zero: # re_zero_var 
                cur_re_zero_var = torch.nn.Parameter(
                    torch.zeros(size=(1,), dtype=torch.float32, requires_grad=True), requires_grad=True)
                # 
                self.fc_re_zero_vars.append(cur_re_zero_var)
            if self.dropout_rate:
                cur_dropout_layer = nn.Dropout(p=self.dropout_rate)
                self.fc_dropout_layers.append(cur_dropout_layer)

        if self.layer_norm:
            self.out_layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, inputs, inputs_mask=None):
        ### padding 
        if inputs_mask is None:
            encoder_padding = layer_utils.embedding_to_padding(inputs) # bsz x n_vertices
        else:
            encoder_padding = inputs_mask # inputs_mask: bsz x n_vertices
        bsz = inputs.size(0)
        seq_length = inputs.size(1)
        # encoder_self_attention_bias = layer_utils.attention_bias_ignore_padding(encoder_padding)
        # encoder_self_attention_mask = layer_utils.attention_mask(encoder_padding)
        encoder_self_attention_mask = layer_utils.attention_mask_single_direction(encoder_padding)
        # print(f"in vertex model forwarding function, encoder_self_attention_mask: {encoder_self_attention_mask.size()}, inputs: {inputs.size()}")
        encoder_self_attention_mask = encoder_self_attention_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        encoder_self_attention_mask = encoder_self_attention_mask.contiguous().view(bsz * self.num_heads, seq_length, seq_length).contiguous()

        seq_length = inputs.size(1)
        x = inputs

        # atten_mask = np.tri(seq_length, seq_length, -1.0, dtype=np.float32).T
        # atten_mask = torch.from_numpy(atten_mask).float().cuda()

        for i in range(self.num_layers):
            res = x.clone()
            if self.layer_norm:
                res = self.layer_norm_layers[i](res)
            # print(f"before attention {i}/{self.num_layers}, res: {res.size()}")
            # res, _ = self.attention_layers[i](res, res, res, attn_mask=atten_mask)
            res, _ = self.attention_layers[i](res, res, res, attn_mask=encoder_self_attention_mask)
            # print(f"after attention {i}/{self.num_layers}, res: {res.size()}")
            if self.re_zero:
                res = res * self.re_zero_vars[i]
            if self.dropout_rate:
                res = self.dropout_layers[i](res)
            x = x + res

            res = x.clone()
            if self.layer_norm:
                res = self.fc_layer_norm_layers[i](res)
            res = self.fc_layers[i](res)
            if self.re_zero:
                res = res * self.fc_re_zero_vars[i]
            if self.dropout_rate:
                res = self.fc_dropout_layers[i](res)
            x = x + res
        if self.layer_norm:
            x = self.out_layer_norm(x)
        return x



class TransformerDecoder(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 fc_size=1024,
                 num_heads=4,
                 layer_norm=True,
                 num_layers=8,
                 dropout_rate=0.2,
                 re_zero=True,
                 ):
        super(TransformerDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.fc_size = fc_size
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.re_zero = re_zero

        ### Attention layer and related modules ###
        self.attention_layers = nn.ModuleList()
        if self.layer_norm:
            self.layer_norm_layers = nn.ModuleList()
        if self.re_zero:
            self.re_zero_vars = nn.ParameterList()
        if self.dropout_rate:
            self.dropout_layers = nn.ModuleList()
        for i in range(self.num_layers):
            cur_atten_layer = nn.MultiheadAttention(
                self.hidden_size, self.num_heads, dropout=0.0, bias=True, kdim=self.hidden_size, vdim=self.hidden_size,
                batch_first=True)
            self.attention_layers.append(cur_atten_layer)
            if self.layer_norm:
                cur_layer_norm = nn.LayerNorm(self.hidden_size)
                self.layer_norm_layers.append(cur_layer_norm)
            if self.re_zero:
                cur_re_zero_var = torch.nn.Parameter(torch.zeros(size=(1,), dtype=torch.float32, requires_grad=True), requires_grad=True)
                self.re_zero_vars.append(cur_re_zero_var)
            if self.dropout_rate:
                cur_dropout_layer = nn.Dropout(p=self.dropout_rate)
                self.dropout_layers.append(cur_dropout_layer)

        ##### attention, re_zero, dropout layers for the context attention layers #####
        self.context_attention_layers = nn.ModuleList()
        if self.layer_norm:
            self.context_norm_layers = nn.ModuleList()
        if self.re_zero:
            self.context_re_zero_vars = nn.ParameterList()
        if self.dropout_rate:
            self.context_dropout_layers = nn.ModuleList()
        for i in range(self.num_layers):
            cur_atten_layer = nn.MultiheadAttention(
                self.hidden_size, self.num_heads, dropout=0.0, bias=True, kdim=self.hidden_size, vdim=self.hidden_size,
                batch_first=True)
            self.context_attention_layers.append(cur_atten_layer)
            if self.layer_norm:
                cur_layer_norm = nn.LayerNorm(self.hidden_size)
                self.context_norm_layers.append(cur_layer_norm)
            if self.re_zero:
                cur_re_zero_var = torch.nn.Parameter(torch.zeros(size=(1,), dtype=torch.float32, requires_grad=True), requires_grad=True)
                self.context_re_zero_vars.append(cur_re_zero_var)
            if self.dropout_rate:
                cur_dropout_layer = nn.Dropout(p=self.dropout_rate)
                # dropout layers
                self.context_dropout_layers.append(cur_dropout_layer)

        ### Attention layer and related modules ###
        self.fc_layers = nn.ModuleList()
        if self.layer_norm:
            self.fc_layer_norm_layers = nn.ModuleList()
        if self.re_zero:
            self.fc_re_zero_vars = nn.ParameterList()
            # self.fc_re_zero_vars = nn.ModuleList()
        if self.dropout_rate:
            self.fc_dropout_layers = nn.ModuleList()
        for i in range(self.num_layers):
            cur_fc_layer = nn.Linear(in_features=self.hidden_size, out_features=self.fc_size, bias=True)
            cur_fc_layer_2 = nn.Linear(in_features=self.fc_size, out_features=self.hidden_size, bias=True)
            self.fc_layers.append(nn.Sequential(*[cur_fc_layer, cur_fc_layer_2]))
            if self.layer_norm:
                cur_layer_norm = nn.LayerNorm(self.hidden_size)
                self.fc_layer_norm_layers.append(cur_layer_norm)
            if self.re_zero:
                cur_re_zero_var = torch.nn.Parameter(
                    torch.zeros(size=(1,), dtype=torch.float32, requires_grad=True), requires_grad=True)
                self.fc_re_zero_vars.append(cur_re_zero_var)
            if self.dropout_rate:
                cur_dropout_layer = nn.Dropout(p=self.dropout_rate)
                self.fc_dropout_layers.append(cur_dropout_layer)

        if self.layer_norm:
            self.out_layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, inputs, sequential_context_embeddings=None, mask=None, sequential_context_mask=None): 
        # mask: mask out padding items
        # sequential context mask: mask out padding items
        seq_length = inputs.size(1)
        bsz = inputs.size(0)

        atten_mask = np.tri(seq_length, seq_length, -1.0, dtype=np.float32).T # tri
        # atten_mask = np.tri(seq_length, seq_length, 0.0, dtype=np.float32)
        atten_mask = torch.from_numpy(atten_mask).float().cuda()
        atten_mask = (atten_mask > 0.5)

        if sequential_context_embeddings is not None:
            context_length = sequential_context_embeddings.size(1)
            # print(f"in decoder's forward function, inputs: {inputs.size()}, sequential_context_embeddings: {sequential_context_embeddings.size()}")
            if sequential_context_mask is None:
              sequential_context_padding = layer_utils.embedding_to_padding(sequential_context_embeddings)
            else:
              sequential_context_padding = 1. - sequential_context_mask.float() # sequential context mask?
              # sequential_context_padding = layer_utils.embedding_to_padding(sequential_context_embeddings)

            sequential_context_atten_mask = layer_utils.attention_mask_single_direction(sequential_context_padding, other_len=seq_length)
            # print(f"in decoder's forward function, sequential_context_padding: {sequential_context_padding.size()}, sequential_context_atten_mask: {sequential_context_atten_mask.size()}")
            sequential_context_atten_mask = sequential_context_atten_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            sequential_context_atten_mask = sequential_context_atten_mask.contiguous().view(bsz * self.num_heads, seq_length, context_length).contiguous()
        
        x = inputs

        for i in range(self.num_layers):
            res = x.clone()
            if self.layer_norm:
                res = self.layer_norm_layers[i](res)
            res, _ = self.attention_layers[i](res, res, res, attn_mask=atten_mask)
            if self.re_zero: # re_zero_vars for the re_zero parameter?
                res = res * self.re_zero_vars[i].unsqueeze(0).unsqueeze(0)
            if self.dropout_rate:
                res = self.dropout_layers[i](res)
            x = x + res

            # if we use sequential context embeddings
            if sequential_context_embeddings is not None:
                # for sequential context embedding
                res = x.clone()
                # then layer_norm, attention layer, re_zero layer and the dropout layer
                if self.layer_norm:
                    res = self.context_norm_layers[i](res)
                res, _ = self.context_attention_layers[i](res, sequential_context_embeddings, sequential_context_embeddings, attn_mask=sequential_context_atten_mask)
                if self.re_zero:
                    res = res * self.context_re_zero_vars[i].unsqueeze(0).unsqueeze(0)
                if self.dropout_rate:
                    res = self.context_dropout_layers[i](res)
                x = x + res
            

            res = x.clone()
            if self.layer_norm:
                res = self.fc_layer_norm_layers[i](res)
            res = self.fc_layers[i](res)
            if self.re_zero:
                res = res * self.fc_re_zero_vars[i]
            if self.dropout_rate: # dropout layers # fc_dropout_layers
                res = self.fc_dropout_layers[i](res)
            x = x + res
        if self.layer_norm:
            x = self.out_layer_norm(x)
        return x


class AdapterTuning(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 down_r_size=128,
                 num_layers=8,
                 ):
        super(AdapterTuning, self).__init__()
        self.adapter_modules = nn.ModuleList()
        for i_layer in range(num_layers):
            cur_layer_adapter_module = nn.Sequential(
                nn.Linear(in_features=hidden_size, out_features=down_r_size, bias=True),
                nn.ReLU(),
                nn.Linear(in_features=down_r_size, out_features=hidden_size, bias=True)
            )
            self.adapter_modules.append(cur_layer_adapter_module)
    def forward_kth_layer(self, h, k):
        # print(f"k {k}, lenght: {len(self.adapter_modules)}")
        delta_h = self.adapter_modules[k](h)
        h = h + delta_h
        return h

class PrefixKeyValues(nn.Module):
    def __init__(self, 
                 hidden_size=256,
                 prefix_key_len=16,
                 prefix_value_len=16,
                 num_layers=8):
        super(PrefixKeyValues, self).__init__()
        self.prefix_key_len = prefix_key_len
        self.prefix_value_len = prefix_value_len
        self.hidden_size = hidden_size
        self.prefix_keys = nn.ParameterList() # prefix keys
        self.prefix_values = nn.ParameterList() # prefix values

        for i_layer in range(num_layers):
            cur_prefix_key_vars = torch.nn.Parameter(torch.zeros(size=(1, self.prefix_key_len, self.hidden_size), dtype=torch.float32, requires_grad=True), requires_grad=True)
            cur_prefix_value_vars = torch.nn.Parameter(torch.zeros(size=(1, self.prefix_value_len, self.hidden_size), dtype=torch.float32, requires_grad=True), requires_grad=True)
            self.prefix_keys.append(cur_prefix_key_vars)
            self.prefix_values.append(cur_prefix_value_vars)
        # 

class PromptValues(nn.Module):
    def __init__(self, 
                 hidden_size=256,
                 prefix_key_len=16,
                 prefix_value_len=16,
                 num_layers=8,
                 nn_prompts=3):
        super(PromptValues, self).__init__()
        self.prefix_key_len = prefix_key_len
        self.prefix_value_len = prefix_value_len
        
        self.hidden_size = hidden_size
        # self.prefix_keys = nn.ParameterList() # prefix keys
        # self.prefix_values = nn.ParameterList() # prefix values
        self.prompt_values = nn.ParameterList()

        for i_layer in range(nn_prompts):
            cur_prefix_key_vars = torch.nn.Parameter(torch.zeros(size=(1, self.prefix_key_len, self.hidden_size), dtype=torch.float32, requires_grad=True), requires_grad=True)
            # cur_prefix_value_vars = torch.nn.Parameter(torch.zeros(size=(1, self.prefix_value_len, self.hidden_size), dtype=torch.float32, requires_grad=True), requires_grad=True)
            self.prompt_values.append(cur_prefix_key_vars)
            # self.prefix_values.append(cur_prefix_value_vars)

# ### Attention layer and related modules ###
        # self.attention_layers = nn.ModuleList()
        # if self.layer_norm:
        #     self.layer_norm_layers = nn.ModuleList()
        # if self.re_zero:
        #     self.re_zero_vars = nn.ParameterList()
        #     # self.re_zero_vars = nn.ModuleList()
        #     # self.re_zero_vars = nn.Paramter
        # if self.dropout_rate:
        #     self.dropout_layers = nn.ModuleList()
        
        # for i in range(self.num_layers):
        #     cur_atten_layer = nn.MultiheadAttention(
        #         self.hidden_size, self.num_heads, dropout=0.0, bias=True, kdim=self.hidden_size, vdim=self.hidden_size,
        #         batch_first=True)
        #     self.attention_layers.append(cur_atten_layer)
        #     if self.layer_norm:
        #         cur_layer_norm = nn.LayerNorm(self.hidden_size)
        #         self.layer_norm_layers.append(cur_layer_norm)
        #     if self.re_zero:
        #         cur_re_zero_var = torch.nn.Parameter(torch.zeros(size=(1,), dtype=torch.float32, requires_grad=True), requires_grad=True)
        #         self.re_zero_vars.append(cur_re_zero_var)
        #     if self.dropout_rate:
        #         cur_dropout_layer = nn.Dropout(p=self.dropout_rate)
        #         self.dropout_layers.append(cur_dropout_layer)

class PromptValuesPrompt(nn.Module):
    def __init__(self, 
                 hidden_size=256,
                 fc_size=512,
                 prefix_key_len=16,
                 prefix_value_len=16,
                 num_layers=3, # we use three attention layers?
                 nn_prompts=3,
                 num_heads=4,
                 layer_norm=True,
                 re_zero=True,
                 dropout_rate=0.4):
        super(PromptValuesPrompt, self).__init__()
        self.prefix_key_len = prefix_key_len
        self.prefix_value_len = prefix_value_len
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.fc_size = fc_size
        
        self.hidden_size = hidden_size
        self.layer_norm = layer_norm
        self.re_zero = re_zero
        self.dropout_rate = dropout_rate
        # self.prefix_keys = nn.ParameterList() # prefix keys
        # self.prefix_values = nn.ParameterList() # prefix values
        self.prompt_values = nn.ParameterList()
        self.prompt_attention_layers = nn.ModuleList()
        self.prompt_layer_norm_layers = nn.ModuleList()
        self.prompt_rezero_vars = nn.ParameterList()
        self.dropout_layers = nn.ModuleList()
        self.prompt_out_layer_norm_layers = nn.ModuleList()

        self.prompt_fc_layers = nn.ModuleList()
        self.prompt_fc_layer_norm_layers = nn.ModuleList()
        self.prompt_fc_rezero_vars = nn.ParameterList()
        self.prompt_fc_dropout_layers = nn.ModuleList()

        for i_prompt in range(nn_prompts):
            cur_prefix_key_vars = torch.nn.Parameter(torch.zeros(size=(1, self.prefix_key_len, self.hidden_size), dtype=torch.float32, requires_grad=True), requires_grad=True)
            # cur_prefix_value_vars = torch.nn.Parameter(torch.zeros(size=(1, self.prefix_value_len, self.hidden_size), dtype=torch.float32, requires_grad=True), requires_grad=True)
            self.prompt_values.append(cur_prefix_key_vars)
            # self.prefix_values.append(cur_prefix_value_vars)

            cur_prompt_attenion_layers = nn.ModuleList()
            cur_prompt_layer_norm_layers = nn.ModuleList()
            # cur_prompt_re_zero_layers = nn.ParameterList()
            cur_prompt_dropout_layers = nn.ModuleList()

            cur_prompt_fc_layers = nn.ModuleList()
            cur_prompt_fc_layer_norm_layers = nn.ModuleList()
            # cur_prompt_fc_re_zero_layers = nn.ParameterList()
            cur_prompt_fc_dropout_layers = nn.ModuleList()

            ##### prompt values #####
            for i_layer in range(num_layers):
                cur_atten_layer = nn.MultiheadAttention(
                self.hidden_size, self.num_heads, dropout=0.0, bias=True, kdim=self.hidden_size, vdim=self.hidden_size,
                batch_first=True)
                # self.attention_layers.append(cur_atten_layer)
                cur_prompt_attenion_layers.append(cur_atten_layer)
                if self.layer_norm:
                    cur_layer_norm = nn.LayerNorm(self.hidden_size)
                    # self.layer_norm_layers.append(cur_layer_norm)
                    cur_prompt_layer_norm_layers.append(cur_layer_norm)
                if self.re_zero:
                    cur_re_zero_var = torch.nn.Parameter(torch.zeros(size=(1,), dtype=torch.float32, requires_grad=True), requires_grad=True)
                    # cur_prompt_re_zero_layers.append(cur_re_zero_var)
                    self.prompt_rezero_vars.append(cur_re_zero_var)
                if self.dropout_rate:
                    cur_dropout_layer = nn.Dropout(p=self.dropout_rate)
                    cur_prompt_dropout_layers.append(cur_dropout_layer)

                cur_fc_layer = nn.Linear(in_features=self.hidden_size, out_features=self.fc_size, bias=True)
                cur_fc_layer_2 = nn.Linear(in_features=self.fc_size, out_features=self.hidden_size, bias=True)
                cur_prompt_fc_layers.append(nn.Sequential(*[cur_fc_layer, cur_fc_layer_2]))
                if self.layer_norm:
                    cur_layer_norm = nn.LayerNorm(self.hidden_size)
                    cur_prompt_fc_layer_norm_layers.append(cur_layer_norm)
                if self.re_zero:
                    cur_re_zero_var = torch.nn.Parameter(
                        torch.zeros(size=(1,), dtype=torch.float32, requires_grad=True), requires_grad=True)
                    # cur_prompt_fc_re_zero_layers.append(cur_re_zero_var)
                    self.prompt_fc_rezero_vars.append(cur_re_zero_var)
                if self.dropout_rate:
                    cur_dropout_layer = nn.Dropout(p=self.dropout_rate)
                    cur_prompt_fc_dropout_layers.append(cur_dropout_layer)

            self.prompt_attention_layers.append(cur_prompt_attenion_layers)
            self.prompt_layer_norm_layers.append(cur_prompt_layer_norm_layers)
            # self.prompt_rezero_vars.append(cur_prompt_re_zero_layers)
            self.dropout_layers.append(cur_prompt_dropout_layers)

            self.prompt_fc_layers.append(cur_prompt_fc_layers)
            self.prompt_fc_layer_norm_layers.append(cur_prompt_fc_layer_norm_layers)
            # self.prompt_fc_rezero_vars.append(cur_prompt_fc_re_zero_layers)
            self.prompt_fc_dropout_layers.append(cur_prompt_fc_dropout_layers)

            cur_prompt_out_layer_norm_layers = nn.LayerNorm(self.hidden_size)
            self.prompt_out_layer_norm_layers.append(cur_prompt_out_layer_norm_layers)
            ##### prompt values and relevant layers #####

    def apply_attention(self, i_prompt, query_prompt_values, key_value_prompt_values):
        cur_prompt_attention_layers = self.prompt_attention_layers[i_prompt]
        cur_prompt_layer_norm_layers = self.prompt_layer_norm_layers[i_prompt]
        cur_prompt_rezero_vars = self.prompt_rezero_vars[i_prompt * self.num_layers: (i_prompt + 1) * self.num_layers]
        cur_prompt_dropout_layers = self.dropout_layers[i_prompt]
        cur_prompt_out_layer_norm_layers = self.prompt_out_layer_norm_layers[i_prompt]
        
        cur_prompt_fc_layers = self.prompt_fc_layers[i_prompt]
        cur_prompt_fc_layer_norm_layers = self.prompt_fc_layer_norm_layers[i_prompt]
        cur_prompt_fc_re_zero_layers = self.prompt_fc_rezero_vars[i_prompt * self.num_layers: (i_prompt + 1) * self.num_layers]
        cur_prompt_fc_dropout_layers = self.prompt_fc_dropout_layers[i_prompt]

        # atten_mask = None
        x = query_prompt_values.contiguous().transpose(1, 0).contiguous()
        # # x = cat([key_value_prompt_values, x], dim=1) --> prefix_key_len x (n_source + 1) x embedding_dim

        # x = torch.cat(
        #     [key_value_prompt_values, x], dim=1
        # )

        # y = key_value_prompt_values
        for i in range(self.num_layers):
            # for i in range(self.num_layers): # num_layers
            res = x.clone()
            if self.layer_norm: # 
                res = cur_prompt_layer_norm_layers[i](res)
            # print(f"{i}-th layer, res: {res.size()}, key_value_prompt_values: {key_value_prompt_values.size()}, ")
            res, _ = cur_prompt_attention_layers[i](res, key_value_prompt_values, key_value_prompt_values)
            # res, _ = cur_prompt_attention_layers[i](res, res, res)
            if self.re_zero:
                res = res * cur_prompt_rezero_vars[i].unsqueeze(0).unsqueeze(0)
            if self.dropout_rate:
                res = cur_prompt_dropout_layers[i](res)
            x = x + res

            res = x.clone()
            if self.layer_norm:
                res = cur_prompt_fc_layer_norm_layers[i](res)
            res = cur_prompt_fc_layers[i](res)
            if self.re_zero:
                res = res * cur_prompt_fc_re_zero_layers[i]
            if self.dropout_rate: # dropout layers # fc_dropout_layers
                res = cur_prompt_fc_dropout_layers[i](res)
            x = x + res
        if self.layer_norm:
            x = cur_prompt_out_layer_norm_layers(x)
        x = x[:, -1:]
        return x
            
    def forward(self, tot_soruce_prompt_values):
        # source_prompt_values: n_source x prefix_key_len x embedding_dim
        # 
        # prefix_key_len: source,
        # tot_prompt_values = nn.ParameterList
        tot_prompt_values = []
        for i_prompt, source_prompt_values in enumerate(tot_soruce_prompt_values):
            # source_prompt_values = 

            source_prompt_values = source_prompt_values.contiguous().transpose(1, 0).contiguous() # prefix_key_len x n_source x embedding_dim
            query_prompt_values = self.prompt_values[i_prompt] # query_prompt_values: prefix_key_len x embedding_dim
            # query_prompt_values = query_prompt_values.unsqueeze(1) # query_prompt_values: prefix_key_len x 1 x embedding_dim
            
            cur_prompt_values = self.apply_attention(i_prompt, query_prompt_values, source_prompt_values)
            tot_prompt_values.append(cur_prompt_values.squeeze(1))

            # cur_prompt_values = self.prompt_values[i_prompt]

            # tot_prompt_values.append(cur_prompt_values.squeeze(0))
            # print(f"{i_prompt}-th prompt, size: {cur_prompt_values.squeeze(1).size()}")
        return tot_prompt_values

            

