# import sonnet as snt
# from tensor2tensor.layers import common_attention
# from tensor2tensor.layers import common_layers
# import tensorflow.compat.v1 as tf
# from tensorflow.python.framework import function
# import tensorflow_probability as tfp

from cmath import log
from tracemalloc import is_tracing
from turtle import reset
import numpy as np
import torch.nn as nn
import layer_utils
import torch
import data_utils_torch as data_utils
import math

from options.options import opt



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
        # self.num_heads = 1
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
                 with_seq_context=False
                 ):
        super(TransformerDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.fc_size = fc_size
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.re_zero = re_zero
        self.with_seq_context = with_seq_context
        self.context_window = opt.model.context_window
        self.atten_mask = None
        self.context_atten_mask = None

        ### Attention layer and related modules ###
        self.attention_layers = nn.ModuleList()
        if self.layer_norm:
            self.layer_norm_layers = nn.ModuleList()
        if self.re_zero:
            self.re_zero_vars = nn.ParameterList()
            # self.re_zero_vars = nn.ModuleList()
            # self.re_zero_vars = nn.Paramter
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
        
        if self.with_seq_context:
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
        seq_length = inputs.size(1)
        bsz = inputs.size(0)
        

        # print(f"inputs: {inputs.size()}")

        if self.training:
            if self.atten_mask is None:
                atten_mask = np.tri(seq_length, seq_length, -1.0, dtype=np.float32).T # tri
                # atten_mask = np.tri(seq_length, seq_length, 0.0, dtype=np.float32)
                atten_mask = torch.from_numpy(atten_mask).float().cuda()
                self.atten_mask = atten_mask
            else:
                atten_mask = self.atten_mask
        else:
            atten_mask = np.tri(seq_length, seq_length, -1.0, dtype=np.float32).T # tri
            # atten_mask = np.tri(seq_length, seq_length, 0.0, dtype=np.float32)
            atten_mask = torch.from_numpy(atten_mask).float().cuda()

        # context_window 
        if self.context_window > 0 and sequential_context_embeddings is None:
            # ##### add global context embeddings to embedding vectors ##### #
            # inputs = inputs[:, 0:1] + inputs # add the contextual information to inputs # not add...
            # if opt.model.debug:
            #     print(f"Using context window {self.context_window} for decoding...")
            if self.training:
                if self.context_atten_mask is None:
                    context_atten_mask = np.tri(seq_length, seq_length, -1.0 * float(self.context_window), dtype=np.float32)
                    context_atten_mask = torch.from_numpy(context_atten_mask).float().cuda()
                    self.context_atten_mask = context_atten_mask
                else:
                    context_atten_mask = self.context_atten_mask
            else:
                context_atten_mask = np.tri(seq_length, seq_length, -1.0 * float(self.context_window), dtype=np.float32)
                context_atten_mask = torch.from_numpy(context_atten_mask).float().cuda()
            atten_mask = context_atten_mask + atten_mask
        # context attention mask
        atten_mask = (atten_mask > 0.5)

        

        # print(atten_mask)

        if sequential_context_embeddings is not None:
            context_length = sequential_context_embeddings.size(1)

            # sequential_context_padding = layer_utils.embedding_to_padding(sequential_context_embeddings)

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
            if self.re_zero:
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

class TransformerDecoderGridBak(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 fc_size=1024,
                 num_heads=4,
                 layer_norm=True,
                 num_layers=8,
                 dropout_rate=0.2,
                 re_zero=True,
                 with_seq_context=False
                 ):
        super(TransformerDecoderGridBak, self).__init__()
        self.hidden_size = hidden_size
        self.fc_size = fc_size
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.re_zero = re_zero
        self.with_seq_context = with_seq_context
        self.context_window = opt.model.context_window
        self.atten_mask = None
        self.context_atten_mask = None
        # attention layer and related modules #
        # only positional encoding and 
        # [grid positional encoding, grid coordinate embedding, grid semantic embedding]
        # [grid positional encoding, grid coordinate embedding] () --> aware of current grid's position, coordinate, and previous grids' semantic embeddings --> for current grid's semantics decoding

        # #### query = [positional encoding, coordinate embedding]; key/value = [positional encoding, coordinate embedding, semantic embedding], not know self (for grid semantic information decoding)
        # #### query = [positional encoding]; key/value = [positional encoding, coordinate embedding, semantic embedding], not know self (for grid coordinate embedding decoding)

        # [grid positional encoding, grid coordinate embedding, grid semantic embedding] --> aware of no information w.r.t. current grid --> for current grid's 
        # [grid semantic encoding + grid positional encoding]; or we just 
        # grid positional encoding;
        # grid semantic encoding

        ### Attention layer and related modules ###
        self.attention_layers = nn.ModuleList()
        if self.layer_norm:
            self.layer_norm_layers = nn.ModuleList()
        if self.re_zero:
            self.re_zero_vars = nn.ParameterList()
            # self.re_zero_vars = nn.ModuleList()
            # self.re_zero_vars = nn.Paramter
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
        
        if self.with_seq_context:
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
        # inputs: total inputs; sequential_context_embeddings: sequential context embedding
        seq_length = inputs.size(1)
        bsz = inputs.size(0) # inputs 
        
        if self.training:
            if self.atten_mask is None:
                # atten_mask = np.tri(seq_length, seq_length, -1.0, dtype=np.float32).T # tri
                atten_mask = np.tri(seq_length, seq_length, 0.0, dtype=np.float32).T # tri
                # atten_mask = np.tri(seq_length, seq_length, 0.0, dtype=np.float32)
                atten_mask = torch.from_numpy(atten_mask).float().cuda()
                self.atten_mask = atten_mask
            else:
                atten_mask = self.atten_mask
        else:
            # atten_mask = np.tri(seq_length, seq_length, -1.0, dtype=np.float32).T # tri
            atten_mask = np.tri(seq_length, seq_length, 0.0, dtype=np.float32).T # tri
            # atten_mask = np.tri(seq_length, seq_length, 0.0, dtype=np.float32)
            atten_mask = torch.from_numpy(atten_mask).float().cuda()

        # # context_window; 
        # if self.context_window > 0 and sequential_context_embeddings is None:
        #     # ##### add global context embeddings to embedding vectors ##### #
        #     # inputs = inputs[:, 0:1] + inputs # add the contextual information to inputs # not add...
        #     # if opt.model.debug:
        #     #     print(f"Using context window {self.context_window} for decoding...")
        #     if self.training:
        #         if self.context_atten_mask is None:
        #             context_atten_mask = np.tri(seq_length, seq_length, -1.0 * float(self.context_window), dtype=np.float32)
        #             context_atten_mask = torch.from_numpy(context_atten_mask).float().cuda()
        #             self.context_atten_mask = context_atten_mask
        #         else:
        #             context_atten_mask = self.context_atten_mask
        #     else:
        #         context_atten_mask = np.tri(seq_length, seq_length, -1.0 * float(self.context_window), dtype=np.float32)
        #         context_atten_mask = torch.from_numpy(context_atten_mask).float().cuda()
        #     atten_mask = context_atten_mask + atten_mask

        # context attention mask
        atten_mask = (atten_mask > 0.5)

        

        # print(atten_mask)

        # if sequential_context_embeddings is not None:
        #     context_length = sequential_context_embeddings.size(1)

        #     # sequential_context_padding = layer_utils.embedding_to_padding(sequential_context_embeddings)

        #     if sequential_context_mask is None:
        #       sequential_context_padding = layer_utils.embedding_to_padding(sequential_context_embeddings)
        #     else:
        #       sequential_context_padding = 1. - sequential_context_mask.float() # sequential context mask?
        #       # sequential_context_padding = layer_utils.embedding_to_padding(sequential_context_embeddings)

        #     sequential_context_atten_mask = layer_utils.attention_mask_single_direction(sequential_context_padding, other_len=seq_length)
        #     # print(f"in decoder's forward function, sequential_context_padding: {sequential_context_padding.size()}, sequential_context_atten_mask: {sequential_context_atten_mask.size()}")
        #     sequential_context_atten_mask = sequential_context_atten_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        #     sequential_context_atten_mask = sequential_context_atten_mask.contiguous().view(bsz * self.num_heads, seq_length, context_length).contiguous()
        
        # res and res --- content

        x = inputs

        x = inputs[:, 1:, :]
        atten_mask = atten_mask[1:, :]

        

        for i in range(self.num_layers): # num_layers
            res = x.clone()
            if self.layer_norm:
                res = self.layer_norm_layers[i](res)
            res, _ = self.attention_layers[i](res, sequential_context_embeddings, sequential_context_embeddings, attn_mask=atten_mask)
            if self.re_zero:
                res = res * self.re_zero_vars[i].unsqueeze(0).unsqueeze(0)
            if self.dropout_rate:
                res = self.dropout_layers[i](res)
            x = x + res

            # # if we use sequential context embeddings
            # if sequential_context_embeddings is not None:
            #     # for sequential context embedding
            #     res = x.clone()
            #     # then layer_norm, attention layer, re_zero layer and the dropout layer
            #     if self.layer_norm:
            #         res = self.context_norm_layers[i](res)
            #     res, _ = self.context_attention_layers[i](res, sequential_context_embeddings, sequential_context_embeddings, attn_mask=sequential_context_atten_mask)
            #     if self.re_zero:
            #         res = res * self.context_re_zero_vars[i].unsqueeze(0).unsqueeze(0)
            #     if self.dropout_rate:
            #         res = self.context_dropout_layers[i](res)
            #     x = x + res
            

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
        # not self embedding
        return x



class TransformerDecoderGrid(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 fc_size=1024,
                 num_heads=4,
                 layer_norm=True,
                 num_layers=8,
                 dropout_rate=0.2,
                 re_zero=True,
                 with_seq_context=False,
                 ):
        super(TransformerDecoderGrid, self).__init__()
        self.hidden_size = hidden_size
        self.fc_size = fc_size
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.re_zero = re_zero
        self.with_seq_context = with_seq_context
        self.context_window = opt.model.context_window
        self.atten_mask = None
        self.context_atten_mask = None
        self.prefix_key_len = opt.model.prefix_key_len
        # attention layer and related modules #
        # only positional encoding and 
        # [grid positional encoding, grid coordinate embedding, grid semantic embedding]
        # [grid positional encoding, grid coordinate embedding] () --> aware of current grid's position, coordinate, and previous grids' semantic embeddings --> for current grid's semantics decoding

        # #### query = [positional encoding, coordinate embedding]; key/value = [positional encoding, coordinate embedding, semantic embedding], not know self (for grid semantic information decoding)
        # #### query = [positional encoding]; key/value = [positional encoding, coordinate embedding, semantic embedding], not know self (for grid coordinate embedding decoding)

        # [grid positional encoding, grid coordinate embedding, grid semantic embedding] --> aware of no information w.r.t. current grid --> for current grid's 
        # [grid semantic encoding + grid positional encoding]; or we just 
        # grid positional encoding;
        # grid semantic encoding

        ### Attention layer and related modules ###
        self.attention_layers = nn.ModuleList()
        if self.layer_norm:
            self.layer_norm_layers = nn.ModuleList()
        if self.re_zero:
            self.re_zero_vars = nn.ParameterList()
            # self.re_zero_vars = nn.ModuleList()
            # self.re_zero_vars = nn.Paramter
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
        
        if self.with_seq_context:
            # if self.with_seq_context:
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

    def forward(self, inputs, sequential_context_embeddings=None, mask=None, sequential_context_mask=None, context_window=None):
        # inputs: total inputs; sequential_context_embeddings: sequential context embedding
        context_window = opt.model.context_window if context_window is None else context_window
        seq_length = inputs.size(1)
        bsz = inputs.size(0) # inputs
        
        # seq_length x seq_length --> attentiion
        atten_mask = np.tri(seq_length, seq_length, -1.0, dtype=np.float32).T # tri
        # atten_mask = np.tri(seq_length, seq_length, 0.0, dtype=np.float32)
        atten_mask = torch.from_numpy(atten_mask).float().cuda()
        self.atten_mask = atten_mask

        if context_window > 0: # context window # a positive context dinwod
            context_atten_mask = np.tri(seq_length, seq_length, -1.0 * float(context_window), dtype=np.float32)
            context_atten_mask = torch.from_numpy(context_atten_mask).float().cuda()
            atten_mask = context_atten_mask + atten_mask

        # context attention mask
        atten_mask = (atten_mask > 0.5)
        atten_mask[:, :self.prefix_key_len] = False
        
        if sequential_context_embeddings is not None:
            sequential_context_embeddings = sequential_context_embeddings[:, 1:]
            seq_atten_mask = atten_mask.clone()
            seq_atten_mask = seq_atten_mask[:, :-1]


        # res and res --- content

        x = inputs

        # x = inputs[:, 1:, :]
        # atten_mask = atten_mask[1:, :]

        for i in range(self.num_layers): # num_layers
            res = x.clone()
            if self.layer_norm:
                res = self.layer_norm_layers[i](res)
            res, _ = self.attention_layers[i](res, res, res, attn_mask=atten_mask)
            if self.re_zero:
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
                res, _ = self.context_attention_layers[i](res, sequential_context_embeddings, sequential_context_embeddings, attn_mask=seq_atten_mask)
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
        # not self embedding
        # x = x[:, :-1]
        x = x[:, self.prefix_key_len - 1: -1]
        return x

class TransformerDecoderInnerGrid(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 fc_size=1024,
                 num_heads=4,
                 layer_norm=True,
                 num_layers=8,
                 dropout_rate=0.2,
                 re_zero=True,
                 with_seq_context=False
                 ):
        super(TransformerDecoderInnerGrid, self).__init__()
        self.hidden_size = hidden_size
        self.fc_size = fc_size
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.re_zero = re_zero
        self.with_seq_context = with_seq_context
        self.context_window = opt.model.context_window
        self.atten_mask = None
        self.context_atten_mask = None
        # attention layer and related modules #
        # only positional encoding and 
        # [grid positional encoding, grid coordinate embedding, grid semantic embedding]
        # [grid positional encoding, grid coordinate embedding] () --> aware of current grid's position, coordinate, and previous grids' semantic embeddings --> for current grid's semantics decoding

        # #### query = [positional encoding, coordinate embedding]; key/value = [positional encoding, coordinate embedding, semantic embedding], not know self (for grid semantic information decoding)
        # #### query = [positional encoding]; key/value = [positional encoding, coordinate embedding, semantic embedding], not know self (for grid coordinate embedding decoding)

        # [grid positional encoding, grid coordinate embedding, grid semantic embedding] --> aware of no information w.r.t. current grid --> for current grid's 
        # [grid semantic encoding + grid positional encoding]; or we just 
        # grid positional encoding;
        # grid semantic encoding

        ### Attention layer and related modules ###
        self.attention_layers = nn.ModuleList()
        if self.layer_norm:
            self.layer_norm_layers = nn.ModuleList()
        if self.re_zero:
            self.re_zero_vars = nn.ParameterList()
            # self.re_zero_vars = nn.ModuleList()
            # self.re_zero_vars = nn.Paramter
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
        
        # if self.with_seq_context:
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
        # inputs: total inputs; sequential_context_embeddings: sequential context embedding
        seq_length = inputs.size(1)
        bsz = inputs.size(0) # inputs 
        
        # seq_length x seq_length --> attentiion
        atten_mask = np.tri(seq_length, seq_length, -1.0, dtype=np.float32).T # tri
        # atten_mask = np.tri(seq_length, seq_length, 0.0, dtype=np.float32)
        atten_mask = torch.from_numpy(atten_mask).float().cuda()
        self.atten_mask = atten_mask

        # # context_window; 
        # if self.context_window > 0 and sequential_context_embeddings is None:
        #     # ##### add global context embeddings to embedding vectors ##### #
        #     # inputs = inputs[:, 0:1] + inputs # add the contextual information to inputs # not add...
        #     # if opt.model.debug:
        #     #     print(f"Using context window {self.context_window} for decoding...")
        #     if self.training:
        #         if self.context_atten_mask is None:
        #             context_atten_mask = np.tri(seq_length, seq_length, -1.0 * float(self.context_window), dtype=np.float32)
        #             context_atten_mask = torch.from_numpy(context_atten_mask).float().cuda()
        #             self.context_atten_mask = context_atten_mask
        #         else:
        #             context_atten_mask = self.context_atten_mask
        #     else:
        #         context_atten_mask = np.tri(seq_length, seq_length, -1.0 * float(self.context_window), dtype=np.float32)
        #         context_atten_mask = torch.from_numpy(context_atten_mask).float().cuda()
        #     atten_mask = context_atten_mask + atten_mask

        # context attention mask
        atten_mask = (atten_mask > 0.5)
        
        # if sequential_context_embeddings is not None:
        #     sequential_context_embeddings = sequential_context_embeddings[:, 1:]
        #     seq_atten_mask = atten_mask.clone()
        #     seq_atten_mask = seq_atten_mask[:, :-1]


        # res and res --- content

        x = inputs

        # x = inputs[:, 1:, :]
        # atten_mask = atten_mask[1:, :]

        

        for i in range(self.num_layers): # num_layers
            res = x.clone()
            if self.layer_norm:
                res = self.layer_norm_layers[i](res)
            res, _ = self.attention_layers[i](res, res, res, attn_mask=atten_mask)
            if self.re_zero:
                res = res * self.re_zero_vars[i].unsqueeze(0).unsqueeze(0)
            if self.dropout_rate:
                res = self.dropout_layers[i](res)
            x = x + res

            # if we use sequential context embeddings
            # if sequential_context_embeddings is not None:
            #     # for sequential context embedding
            #     res = x.clone()
            #     # then layer_norm, attention layer, re_zero layer and the dropout layer
            #     if self.layer_norm:
            #         res = self.context_norm_layers[i](res)
            #     res, _ = self.context_attention_layers[i](res, sequential_context_embeddings, sequential_context_embeddings, attn_mask=seq_atten_mask)
            #     if self.re_zero:
            #         res = res * self.context_re_zero_vars[i].unsqueeze(0).unsqueeze(0)
            #     if self.dropout_rate:
            #         res = self.context_dropout_layers[i](res)
            #     x = x + res
            

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
        # not self embedding
        x = x[:, :-1]
        return x


# vertex model and ...
#### autoregressive for grids ####
class VertexModel(nn.Module):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 quantization_bits,
                 class_conditional=False,
                 num_classes=55,
                 max_num_input_verts=2500,
                 use_discrete_embeddings=True,
                 inter_part_auto_regressive=False,
                 predict_joint=False,
                 use_multi_gpu=False
                 ): # moduels 
        super(VertexModel, self).__init__()
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.quantization_bits = quantization_bits
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.max_num_input_verts = max_num_input_verts
        self.use_discrete_embeddings = use_discrete_embeddings
        self.inter_part_auto_regressive = inter_part_auto_regressive
        self.predict_joint = predict_joint
        self.embedding_dim = self.decoder_config['hidden_size']
        decoder_config['with_seq_context'] = False
        self.max_sample_length = max_num_input_verts
        # construct encoders and decoders
        self.max_num_grids = opt.vertex_model.max_num_grids
        self.grid_size = opt.vertex_model.grid_size
        self.num_grids_quantization = (2 ** self.quantization_bits) // self.grid_size # gird_xyzs' range
        
        self.vocab_size = 2 ** (self.grid_size ** 3) + 5
        print(f"Constructing VertexModel with vocab size: {self.vocab_size}, grid size: {self.grid_size}.")

        self.grid_pos_embed_max_num = self.num_grids_quantization ** 3
        self.prefix_key_len = opt.model.prefix_key_len
        self.prefix_value_len = opt.model.prefix_value_len

        self.use_multi_gpu = use_multi_gpu

        # grid embedding layers
        # grid embedding layers
        # self.grid_embedding_layers = nn.ModuleList(
        #   [
        #     # nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim),
        #     nn.Embedding(num_embeddings=self.grid_pos_embed_max_num + 5, embedding_dim=self.embedding_dim), # for grid sequential order embedding
        #     nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim), # for grid xyz's embedding
        #     nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim),
        #     nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim),
        #     nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim),
        #   ]
        # )

        ### for grid content encoding ### # grid content encoding #
        self.grid_embedding_layers = nn.ModuleList(
          [
            nn.Embedding(num_embeddings=self.grid_pos_embed_max_num + 5, embedding_dim=self.embedding_dim), # grid order embedding
            nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim), # for grid xyz's embedding; xyz discrete position encodings
            nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim), # for grid xyz's embedding;
            nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim),
            nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim), # xyz indicator embedding
          ]
        )

        self.grid_content_embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)

        # grid embedding layers; if we treat them as tokens?
        self.grid_coord_embedding_layers = nn.ModuleList(
          [
            nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim), # for grid coordinate embedding
            nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim),
            nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim),
          ]
        )

        # #### grid coord embedding layers #### #

        # grid content --> discrete values
        # self.grid_content_embedding_layer = nn.Embedding(num_embeddings=2, embedding_dim=self.embedding_dim)  
        # grid_size
        ########### Grid content conv layers ###########
        # if self.grid_size == 4:
        #     self.grid_content_conv_layers = nn.Sequential(
        #         # (4 x 4 x 4) -> (3 x 3 x 3)
        #     torch.nn.Conv3d(self.embedding_dim, self.embedding_dim, kernel_size=(3, 3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros',),
        #     # (4 x 4 x 4) -> (3 x 3 x 3)
        #     torch.nn.Conv3d(self.embedding_dim, self.embedding_dim, kernel_size=(3, 3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros',),
        #     # (4 x 4 x 4) -> (3 x 3 x 3)
        #     torch.nn.Conv3d(self.embedding_dim, self.embedding_dim, kernel_size=(2, 2, 2), stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',),
        #     # (3 x 3 x 3) -> (2 x 2 x 2)
        #     torch.nn.Conv3d(self.embedding_dim, self.embedding_dim, kernel_size=(2, 2, 2), stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',),
        #     # (2 x 2 x 2) -> (1 x 1 x 1)
        #     torch.nn.Conv3d(self.embedding_dim, self.embedding_dim, kernel_size=(2, 2, 2), stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',)
        #     )
        # else:
        #     nn_conv_layers = int(math.log2(self.grid_size))
        #     print(f"In vertex model with grid size: {self.grid_size}, conv_layers: {nn_conv_layers}.")
        #     # grid_content_conv_layers_list = \
        #     #     [torch.nn.Conv3d(self.embedding_dim, self.embedding_dim, kernel_size=(3, 3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros',) for _ in range(3)] + \
        #     #         [torch.nn.Conv3d(self.embedding_dim, self.embedding_dim, kernel_size=(2, 2, 2), stride=2, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',) for _ in range(nn_conv_layers)]
        #     grid_content_conv_layers_list = [torch.nn.Conv3d(self.embedding_dim, self.embedding_dim, kernel_size=(2, 2, 2), stride=2, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',) for _ in range(nn_conv_layers)]
        #     # self.grid_content_conv_layers = nn.Sequential(
        #     # # (4 x 4 x 4) -> (3 x 3 x 3)
        #     # *[
        #     #     torch.nn.Conv3d(self.embedding_dim, self.embedding_dim, kernel_size=(2, 2, 2), stride=2, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',) for _ in range(nn_conv_layers)]
        #     # )
        #     self.grid_content_conv_layers = nn.Sequential(
        #         *grid_content_conv_layers_list
        #     )
        # self.grid_content_conv_layers = torch.nn.Conv3d(self.embedding_dim, self.embedding_dim, kernel_size=(3, 3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        ########### Grid content conv layers ###########

        # construct encoders and decoders # decoder; 
        decoder_config['with_seq_context'] = True
        self.decoder_grid_content = TransformerDecoderGrid(**decoder_config) # transfomer decoder
        # 
        decoder_config['with_seq_context'] = False
        self.decoder_grid_coord = TransformerDecoderGrid(**decoder_config)
        
        # self.decoder_inner_grid_content = TransformerDecoderInnerGrid(**decoder_config)
        # project to logits, grid coordinates
        self.grid_project_to_logits = nn.ModuleList(
          [
            nn.Linear(self.embedding_dim, self.num_grids_quantization + 1, bias=True),
            nn.Linear(self.embedding_dim, self.num_grids_quantization + 1, bias=True),
            nn.Linear(self.embedding_dim, self.num_grids_quantization + 1, bias=True),
          ]
        )
        
        # decode grid content from embeded latent vectors
        # self.grid_content_project_to_logits = nn.ModuleList(
        #   [nn.Linear(self.embedding_dim, 2, bias=True) for _ in range(self.grid_size ** 3)]
        # )

        self.grid_content_project_to_logits = nn.Linear(self.embedding_dim, self.vocab_size, bias=True) # project to logits for grid content decoding

        # positional encoding
        # self.grid_positional_encoding = nn.Embedding(
        #     num_embeddings=self.grid_size ** 3, embedding_dim=self.embedding_dim
        # )
        # torch.nn.init.xavier_uniform_(self.grid_positional_encoding.weight)
        

        if self.class_conditional:
          ### classical class conditional embedding layers ###
            self.class_embedding_layer = nn.Parameter(
                torch.zeros((self.num_classes, self.prefix_key_len, self.embedding_dim), requires_grad=True, dtype=torch.float32), requires_grad=True
            )
            self.class_embedding_layer_content = nn.Parameter(
                torch.zeros((self.num_classes, self.prefix_key_len, self.embedding_dim), requires_grad=True, dtype=torch.float32), requires_grad=True
            )
            self.class_embedding_layer_grid = nn.Parameter( # requires_grad
                torch.zeros((self.num_classes, self.prefix_key_len, self.embedding_dim), requires_grad=True, dtype=torch.float32), requires_grad=True
            )

            # self.class_embedd

        # self.inputs_embedding_layers = nn.ModuleList([
        #     nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim), # tf.mod(tf.range(seq_length), 3)
        #     nn.Embedding(num_embeddings=self.max_num_input_verts + 5, embedding_dim=self.embedding_dim), # tf.floordiv(tf.range(seq_length), 3)
        #     nn.Embedding(num_embeddings=2**self.quantization_bits + 2, embedding_dim=self.embedding_dim) # quantized vertices
        # ])
        # for cur_layer in self.inputs_embedding_layers:
        #     torch.nn.init.xavier_uniform_(cur_layer.weight)

        if not self.class_conditional: # class condition # claass condition
            self.zero_embed = nn.Parameter(torch.zeros(size=(1, 1, self.embedding_dim), requires_grad=True, dtype=torch.float32), requires_grad=True)


        # logits prediction, joint dir and joint pvp prediction
        # self.project_to_pointer_inter_part = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        # self.project_to_logits = nn.Linear(self.embedding_dim, 2**self.quantization_bits + 1, bias=True)
        # torch.nn.init.xavier_uniform_(self.project_to_logits.weight)

    def _prepare_context(self, context):
      # - content embedding for each grid --> [gird content, grid coord embeddings, grid order embeddings]
      # - grid coord_order embedding --> [grid coord embeddings, grid order embeddings]
        if self.class_conditional:
            # global_context_embedding = self.class_embedding_layer(context['class_label'])
            # global_context_embedding_content = self.class_embedding_layer_content(context['class_label'])
            # global_context_embedding_grid = self.class_embedding_layer_grid(context['class_label'])
            # global_context_embedding = [global_context_embedding, global_context_embedding_content, global_context_embedding_grid]

            global_context_embedding = self.class_embedding_layer[context['class_label']] # class_label: bsz x 1 --> embedding: bsz x 1 x key_len x embedding_dim
            global_context_embedding_content = self.class_embedding_layer_content[context['class_label']]
            global_context_embedding_grid = self.class_embedding_layer_grid[context['class_label']] # class_label: bsz x 1 --> embedding: bsz x 1 x key_len x embedding_dim
            # print("embeddings", global_context_embedding.size(), global_context_embedding_content.size(), global_context_embedding_grid.size())
            # #### global_context_embedding ####
            global_context_embedding = global_context_embedding.squeeze(1)
            global_context_embedding_content = global_context_embedding_content.squeeze(1)
            global_context_embedding_grid = global_context_embedding_grid.squeeze(1)
            global_context_embedding = [global_context_embedding, global_context_embedding_content, global_context_embedding_grid]
        else:
            global_context_embedding = None
        return  global_context_embedding, None


    #### 
    def _embed_input_grids(self, grid_xyzs, grid_content, grid_pos=None, global_context_embedding=None):
      # grid content: bsz x grid_length --> should convert grids into discrete grid content values in the input
      # grid xyzs: bsz x grid_length x 3
      global_context_embedding, global_context_embedding_content, global_context_embedding_grid = global_context_embedding
      # grid_xyzs: n_grids x 3
      bsz = grid_xyzs.size(0)
      grid_length = grid_xyzs.size(1)
      if grid_pos is None:
        grid_order = torch.arange(grid_length).cuda()
        # grid_order_embedding: 1 x grid_length x embedding_dim
        grid_order_embedding = self.grid_embedding_layers[0](grid_order).unsqueeze(0)
      else:
        # compute grid_order_embeddings
        grid_order_embedding = self.grid_embedding_layers[0](grid_pos) # .unsqueeze(0)
      # 
      nn_grid_xyzs = grid_xyzs.size(2)
      grid_coord_embedding = 0.
      # 
    #   grid_coord_embeddings = []
      for c in range(nn_grid_xyzs):
        cur_grid_coord = grid_xyzs[:, :, c] # bsz x grid_length
        # cur_grid_coord_embedding: bsz x grid_length x embedding_dim
        cur_grid_coord_embedding = self.grid_embedding_layers[c + 1](cur_grid_coord) 
        grid_coord_embedding += cur_grid_coord_embedding # cur_grid_coord_embedding: bsz x grid_
        # grid_coord_embeddings.append(cur_grid_coord_embedding.unsqueeze(2))

      grid_coord_embeddings = []
      for c in range(nn_grid_xyzs):
        cur_grid_coord = grid_xyzs[:, :, c] # bsz x grid_length
        # cur_grid_coord_embedding: bsz x grid_length x embedding_dim
        cur_grid_coord_embedding = self.grid_coord_embedding_layers[c](cur_grid_coord) 
        # grid_coord_embedding += cur_grid_coord_embedding # cur_grid_coord_embedding: bsz x grid_
        grid_coord_embeddings.append(cur_grid_coord_embedding.unsqueeze(2))

      # grid order embedding
      grid_embedding = grid_order_embedding + grid_coord_embedding
      grid_coord_embeddings = torch.cat(grid_coord_embeddings, dim=2)
      
      # # flat_grid_content
      # flat_grid_content = grid_content.contiguous().view(grid_content.size(0) * grid_content.size(1), grid_content.size(2), grid_content.size(3), grid_content.size(4)).contiguous()
      # flat_grid_content_embedding = self.grid_content_embedding_layer(flat_grid_content) # bsz x grid_length x (gs x gs x gs) x embedding_dim
      # flat_grid_content_embedding = flat_grid_content_embedding.contiguous().permute(0, 4, 1, 2, 3).contiguous()
      # flat_grid_content_embedding = self.grid_content_conv_layers(flat_grid_content_embedding) # bsz x grid_length x embedding_dim x 1 x 1 x 1
      # flat_grid_content_embedding = flat_grid_content_embedding.contiguous().squeeze(-1).squeeze(-1).squeeze(-1).contiguous()
      # grid_content_embedding = flat_grid_content_embedding.contiguous().view(bsz, grid_length, self.embedding_dim).contiguous()

      grid_content_embedding = self.grid_content_embedding_layer(grid_content) # grid_content: bsz x grid_length --> grid_content_embedding: bsz x grid_length x embedding_size
      grid_content_embedding = grid_content_embedding + grid_embedding # grid_embedding: bsz x grid_length x embedding_size
      

      grid_xyz = torch.arange(0, 3, dtype=torch.long).cuda()
      grid_xyz_embeddings = self.grid_embedding_layers[-1](grid_xyz) # 3 x embedding_dim
      grid_xyz_embeddings = grid_xyz_embeddings.unsqueeze(0).unsqueeze(0).contiguous().repeat(bsz, grid_length, 1, 1).contiguous() # order_embedding: bsz x 
      # grid order embedding and grid xyz embedding?
      # grid xyz embedding
      
      # grid_order_coord_xyz_embeddings: bsz x grid_length x 3 x embedding_dim
      grid_order_coord_xyz_embeddings = grid_order_embedding.contiguous().repeat(bsz, 1, 1).contiguous().unsqueeze(2) + grid_xyz_embeddings + grid_coord_embeddings
      grid_order_coord_xyz_embeddings = grid_order_coord_xyz_embeddings.contiguous().view(bsz, grid_length * 3, -1).contiguous()
      # grid order coord xyz embeddings # order_coord_xyz; order_coord_embeddings
      grid_order_coord_xyz_embeddings = torch.cat(
        [global_context_embedding, grid_order_coord_xyz_embeddings], dim=1
      )

      # [content + order + coord value] # coord value
      # # only predict grids

      # print(f"global_context_embedding: {global_context_embedding.size()}, grid_content_embedding: {grid_content_embedding.size()}")
      # content embeddings 
      grid_content_embedding = torch.cat([global_context_embedding_content, grid_content_embedding], dim=1)

      grid_embedding = torch.cat([global_context_embedding_grid, grid_embedding], dim=1)
      grid_order_embedding = torch.cat([global_context_embedding, grid_order_embedding.contiguous().repeat(bsz, 1, 1).contiguous()], dim=1)

      
      return grid_order_embedding, grid_embedding, grid_content_embedding, grid_order_coord_xyz_embeddings


    def _create_dist_grid_coord_v2(self, grid_order_coord_xyz_embeddings,  temperature=1., top_k=0, top_p=1.0):
    #   bsz, grid_length = grid_order_embedding.size(0), grid_order_coord_xyz_embeddings.size(1) - 1
      # grid_coord_outputs = self.decoder_grid_coord(grid_order_embedding, sequential_context_embeddings=grid_content_embedding)
      grid_coord_outputs = self.decoder_grid_coord(grid_order_coord_xyz_embeddings, sequential_context_embeddings=None, context_window=opt.model.context_window * 3) # only use coord for coord decoding

      # grid_coord_outputs = grid_coord_outputs[:, 1:, :]
      # grid_coord_outputs: bsz x (grid_length x 3) x embedding_dim
      grid_length = grid_coord_outputs.size(1) // 3
      bsz = grid_coord_outputs.size(0)
      # grid_coord_outputs: bsz x grid_length x 3 x embedding_dim
      grid_coord_outputs = grid_coord_outputs.contiguous().view(bsz, grid_length, 3, -1).contiguous()

      # logits_x: bsz x grid_length x nn_grid_coords
      logits_x, logits_y, logits_z = self.grid_project_to_logits[0](grid_coord_outputs[:, :, 0]), self.grid_project_to_logits[1](grid_coord_outputs[:, :, 1]), self.grid_project_to_logits[2](grid_coord_outputs[:, :, 2])

      logits_x /= temperature; logits_y /= temperature; logits_z /= temperature
      logits_x = layer_utils.top_k_logits(logits_x, top_k)
      logits_x = layer_utils.top_p_logits(logits_x, top_p)

      logits_y = layer_utils.top_k_logits(logits_y, top_k)
      logits_y = layer_utils.top_p_logits(logits_y, top_p)

      logits_z = layer_utils.top_k_logits(logits_z, top_k) # bsz x grid_length x (1 + nn_grid_coord_discretization)
      logits_z = layer_utils.top_p_logits(logits_z, top_p)

      logits_xyz = torch.cat(
        [logits_x.unsqueeze(2), logits_y.unsqueeze(2), logits_z.unsqueeze(2)], dim=2
      )
      # bsz x (grid_length x 3) x (1 + nn_grid_coord_discretization)
      # logits_xyz = logits_xyz.contiguous().view(bsz, grid_length * 3, -1).contiguous()
      # then we 
      cat_dist_grid_xyz = torch.distributions.Categorical(logits=logits_xyz)
      return cat_dist_grid_xyz

    def _create_dist_grid_content(self, grid_embedding, grid_content_embedding, temperature=1., top_k=0, top_p=1.):
      # bsz, grid_length = grid_embedding.size(0), grid_embedding.size(1) - self.prefix_key_len

      # grid_content_outputs: bsz x (1 + grid_length) x embedding_dim
      # grid_content_outputs = self.decoder_grid_content(grid_embedding, sequential_context_embeddings=grid_content_embedding)
      # grid_content_outputs: bsz x grid_length x embedding_dim
      grid_content_outputs = self.decoder_grid_content(grid_content_embedding, sequential_context_embeddings=grid_embedding, context_window=opt.model.context_window)
    
    #   ##### Add grid positional encoding & decoder for grid content #####
    #   grid_positional_encoding = torch.arange(start=0, end=self.grid_size ** 3, step=1, dtype=torch.long).cuda()
    #   grid_positional_encoding = self.grid_positional_encoding(grid_positional_encoding)
    #   flat_grid_content_outputs_for_decoding = torch.cat(
    #     [grid_content_outputs.contiguous().view(bsz * grid_length, 1, -1).contiguous(), grid_positional_encoding.unsqueeze(0).repeat(bsz * grid_length, 1, 1).contiguous()], dim=1
    #   )
    #   flat_grid_content_feat = self.decoder_inner_grid_content(flat_grid_content_outputs_for_decoding) # bsz x grid_length x (len) x embedding_dim
    #   grid_content_feat = flat_grid_content_feat.contiguous().view(bsz, grid_length, -1, self.embedding_dim).contiguous()
    #   ##### Add grid positional encoding & decoder for grid content #####

      # grid_content_outputs = grid_content_outputs[:, 1:]

      # grid_content_outputs: bsz x grid_length x embedding_size --> grid_content_logits: bsz x grid_length x (vocab_size)
      grid_content_logits = self.grid_content_project_to_logits(grid_content_outputs)
      grid_content_logits = grid_content_logits / temperature
      grid_content_logits = layer_utils.top_k_logits(grid_content_logits, top_k)
      grid_content_logits = layer_utils.top_p_logits(grid_content_logits, top_p)

      # grid_values = []
      # for i_x in range(self.grid_size ** 3): # together with grid size
      #   # cur_grid_value_proj_layer: bsz x grid_length x 2
      #   cur_grid_value = self.grid_content_project_to_logits[i_x](grid_content_outputs) # use grid content features for decoding directly
      #   # cur_grid_value = self.grid_content_project_to_logits[i_x](grid_content_feat[:, :, i_x, :]) # bsz x grid_length x embedding_dim -> bsz x grid_length x 2
      #   cur_grid_value /= temperature
      #   # cur_grid_value...
      #   # cur_grdi_value: bsz x grid_length x 2
      #   cur_grid_value = layer_utils.top_k_logits(cur_grid_value, top_k)
      #   cur_grid_value = layer_utils.top_p_logits(cur_grid_value, top_p)
      #   grid_values.append(cur_grid_value.unsqueeze(-2))
      # # grid_values: bsz x grid_length x (grid_size ** 3) x 2
      # grid_values = torch.cat(grid_values, dim=-2) # 

      # grid_values: bsz x grid_length x grid_size x grid_size x grid_size x 2
      # grid_values = grid_values.contiguous().view(bsz, grid_length, self.grid_size, self.grid_size, self.grid_size, 2).contiguous()
      # cat_dist_grid_values = torch.distributions.Categorical(logits=grid_values)
      cat_dist_grid_content = torch.distributions.Categorical(logits=grid_content_logits)
      # return cat_dist_grid_values
      return cat_dist_grid_content


    def forward(self, batch):
        global_context, seq_context = self._prepare_context(batch)
        # [global_context_embedding, global_context_embedding_content, global_context_embedding_grid] = 
        # left_cond, rgt_cond = self._prepare_prediction(global_context) # _create_dist() #
        # print(f"vertices flat: {batch['vertices_flat'].size()}")

        #### vertices embeddings ####
        # pred_dist, outputs = self._create_dist(batch['vertices_flat'][:, :-1], embedding_layers=self.inputs_embedding_layers, project_to_logits_layer=self.project_to_logits, global_context_embedding=global_context, sequential_context_embeddings=seq_context,
        # )

        grid_xyzs = batch['grid_xyzs'] # n_grids x (3) --> grid_xyzs
        grid_content = batch['grid_content_vocab'] # use content_vocab for prediction and predict content_vocab
        if 'grid_pos' in batch:
            grid_pos = batch['grid_pos']
        else:
            grid_pos = None
        
        # bsz, grid_length = grid_xyzs.size(0), grid_xyzs.size(1)
        # grid_order_embedding, grid_embedding, grid_content_embedding: bsz x (1 + grid_length) x embedding_dim
        grid_order_embedding, grid_embedding, grid_content_embedding, grid_order_coord_xyz_embeddings = self._embed_input_grids(grid_xyzs=grid_xyzs,  grid_content=grid_content, grid_pos=grid_pos, global_context_embedding=global_context)

        # pred_dist_grid_xyz = self._create_dist_grid_coord(grid_order_embedding, grid_content_embedding)
        pred_dist_grid_xyz = self._create_dist_grid_coord_v2(grid_order_coord_xyz_embeddings,  temperature=1., top_k=0, top_p=1.0)
        # create dist...
        pred_dist_grid_values = self._create_dist_grid_content(grid_embedding, grid_content_embedding)
        # pred_dist_grid_values = self._create_dist_grid_content_v2(grid_embedding, grid_content_embedding, grid_content, is_training=True)

        # pred_dist_grid_xyz, pred_dist_grid_values = self._create_dist_grid(batch, global_context_embedding=global_context, temperature=1., top_k=0, top_p=1.0)

        return pred_dist_grid_xyz, pred_dist_grid_values

    def _loop_sample(self, loop_idx, samples, embedding_layers, project_to_logits_layer, context_feature, seq_context, temperature, top_k, top_p):
        cat_dist, outputs = self._create_dist(
            samples, embedding_layers=embedding_layers, project_to_logits_layer=project_to_logits_layer, global_context_embedding=context_feature, sequential_context_embeddings=seq_context, temperature=temperature, top_k=top_k, top_p=top_p, rt_logits=True
        )
        next_sample = cat_dist.sample()
        next_sample = next_sample[:, -1:] # next # batch 
        samples = torch.cat([samples, next_sample], dim=1)
        # print(f"in vertex looping sampling, samples: {samples.size()}")
        # loop_sample
        next_sample_probs = outputs[:, -1, :]
        next_sample_probs = data_utils.batched_index_select(values=next_sample_probs, indices=next_sample, dim=1) # next_sample_probs: bsz x 1
        # 
        return loop_idx + 1, samples, outputs, next_sample_probs, next_sample # next_sample_probs --> bsz x 1 --> for their corresponding next sample

    def _loop_sample_grid(self, loop_idx, samples, global_context_embedding=None, sequential_context_embeddings=None, temperature=1., top_k=0, top_p=1.0): # loop sample grids
      grid_xyzs = samples['grid_xyzs'] # bsz x (1 + cur_seq_length) x 3
      grid_content = samples['grid_content_vocab'] # bsz x (1 + cur_seq_length) x (gs x gs x gs)
      grid_pos = samples['grid_pos']

    #   bsz = grid_xyzs.size(0)

      max_num_grids = opt.vertex_model.max_num_grids
      max_num_grids = opt.model.context_window if opt.dataset.use_context_window_as_max else max_num_grids # use context window as max_num_grids

      nn_grids = grid_xyzs.size(1)
      if nn_grids > max_num_grids: # maximum number of grids
        grid_xyzs_in = grid_xyzs[:, nn_grids - max_num_grids:]
        grid_content_in = grid_content[:, nn_grids - max_num_grids: ]
        if opt.dataset.use_context_window_as_max: # use context window as max
            grid_pos_in = grid_pos[:, :max_num_grids]
        else:
            grid_pos_in = grid_pos[:, nn_grids - max_num_grids: ]
      else:
        grid_xyzs_in = grid_xyzs
        grid_content_in = grid_content
        grid_pos_in = grid_pos
      
      ##### sample x ##### #### embed input grids...
      grid_order_embedding, grid_embedding, grid_content_embedding, grid_order_coord_xyz_embeddings = self._embed_input_grids(grid_xyzs=grid_xyzs_in,  grid_content=grid_content_in,  grid_pos=grid_pos_in, global_context_embedding=global_context_embedding)
    #   grid_order_embedding, grid_embedding, grid_content_embedding, grid_order_coord_xyz_embeddings = self._embed_input_grids(grid_xyzs=grid_xyzs_in,  grid_content=grid_content_in, global_context_embedding=global_context_embedding)
    
      # create dist grid coord
    #   pred_dist_grid_xyz = self._create_dist_grid_coord(grid_order_embedding, grid_content_embedding)
      pred_dist_grid_xyz = self._create_dist_grid_coord_v2(grid_order_coord_xyz_embeddings,  temperature=1., top_k=0, top_p=1.0)

      # 
      pred_grid_xyz = pred_dist_grid_xyz.sample() # bsz x (1 + cur_seq_length) x 3
      pred_grid_xyz = pred_grid_xyz[:, -1] # bsz x 3 
      
      pred_grid_x = pred_grid_xyz[:, 0]
      grid_xyzs[:, -1, 0] = pred_grid_x
      grid_xyzs_in[:, -1, 0] = pred_grid_x
      ##### sample x #####

      ##### sample y #####
    #   grid_order_embedding, grid_embedding, grid_content_embedding, grid_order_coord_xyz_embeddings = self._embed_input_grids(grid_xyzs=grid_xyzs,  grid_content=grid_content, global_context_embedding=global_context_embedding)
      grid_order_embedding, grid_embedding, grid_content_embedding, grid_order_coord_xyz_embeddings = self._embed_input_grids(grid_xyzs=grid_xyzs_in,  grid_content=grid_content_in, grid_pos=grid_pos_in,  global_context_embedding=global_context_embedding)
      
    #   pred_dist_grid_xyz = self._create_dist_grid_coord(grid_order_embedding, grid_content_embedding)
      pred_dist_grid_xyz = self._create_dist_grid_coord_v2(grid_order_coord_xyz_embeddings,  temperature=1., top_k=0, top_p=1.0)
      # 
      pred_grid_xyz = pred_dist_grid_xyz.sample() # bsz x (1 + cur_seq_length) x 3
      pred_grid_xyz = pred_grid_xyz[:, -1] # bsz x 3 
      
      pred_grid_y = pred_grid_xyz[:, 1]
      grid_xyzs[:, -1, 1] = pred_grid_y
      grid_xyzs_in[:, -1, 1] = pred_grid_y
      ##### sample y #####

      ##### sample z #####
    #   grid_order_embedding, grid_embedding, grid_content_embedding, grid_order_coord_xyz_embeddings = self._embed_input_grids(grid_xyzs=grid_xyzs,  grid_content=grid_content, global_context_embedding=global_context_embedding)
      grid_order_embedding, grid_embedding, grid_content_embedding, grid_order_coord_xyz_embeddings = self._embed_input_grids(grid_xyzs=grid_xyzs_in,  grid_content=grid_content_in, grid_pos=grid_pos_in,  global_context_embedding=global_context_embedding)
      
    #   pred_dist_grid_xyz = self._create_dist_grid_coord(grid_order_embedding, grid_content_embedding)
      pred_dist_grid_xyz = self._create_dist_grid_coord_v2(grid_order_coord_xyz_embeddings,  temperature=1., top_k=0, top_p=1.0)
      # 
      pred_grid_xyz = pred_dist_grid_xyz.sample() # bsz x (1 + cur_seq_length) x 3
      pred_grid_xyz = pred_grid_xyz[:, -1] # bsz x 3 
      

      pred_grid_z = pred_grid_xyz[:, 2]
      grid_xyzs[:, -1, 2] = pred_grid_z
      grid_xyzs_in[:, -1, 2] = pred_grid_z
      ##### sample z #####


    #   grid_order_embedding, grid_embedding, grid_content_embedding, grid_order_coord_xyz_embeddings = self._embed_input_grids(grid_xyzs=grid_xyzs,  grid_content=grid_content, global_context_embedding=global_context_embedding)
      
      # grid_pos_in: ----> grids pos
      # global content embedding
      grid_order_embedding, grid_embedding, grid_content_embedding, grid_order_coord_xyz_embeddings = self._embed_input_grids(grid_xyzs=grid_xyzs_in,  grid_content=grid_content_in, grid_pos=grid_pos_in,  global_context_embedding=global_context_embedding) 

      ####### grid content v1 for sampling 3######
      pred_dist_grid_values = self._create_dist_grid_content(grid_embedding, grid_content_embedding)
      # 
      pred_grid_values = pred_dist_grid_values.sample()
      pred_grid_values = pred_grid_values[:, -1] # bsz x (gs x gs x gs) # 
    ####### grid content v1 for sampling 3######

        ####### grid content v2 for sampling 3###### --> a sequential sampling # grid content v2
    #   pred_grid_values = self._create_dist_grid_content_v2(grid_embedding, grid_content_embedding, grid_content_in, is_training=False)
      
      ####### grid content v1 for sampling 3######

      grid_content[:, -1] = pred_grid_values
      
      # # grid_xyzs: bsz x (1 + 1 + cur_seq_length) x 3
      # grid_xyzs = torch.cat(
      #   [grid_xyzs, torch.zeros((bsz, 1, 3), dtype=torch.long).cuda()], dim=1
      # )
      # # grid_content: bsz x (1 + 1 + cur_seq_length) x (gs x gs x gs)
      # grid_content = torch.cat(
      #   [grid_content, torch.zeros((bsz, 1, self.grid_size, self.grid_size, self.grid_size), dtype=torch.long).cuda()], dim=1
      # )
      
      samples['grid_xyzs'] = grid_xyzs
      samples['grid_content_vocab'] = grid_content
      samples['grid_pos'] = grid_pos
      return loop_idx + 1, samples


    def _stop_cond(self, samples):
        # print(f"in stop condition, samples: {samples}")
        equal_to_zero = (samples == 0).long() # equal to zero

        accum_equal_zero = torch.sum(equal_to_zero, dim=-1) ##### number of xyzs equal to 0
        # not_stop = (accum_equal_zero == 0).long() # 
        # not_stop = (accum_equal_zero <= 2).long() # not stop
        stop_indicator = (accum_equal_zero == 3).long() # all three euqalt to zero --> stop
        stop_indicator = torch.any(stop_indicator, dim=-1).long() # bsz; for each batch size
        not_stop = torch.sum(1 - stop_indicator).item() > 0
        # not_stop = torch.sum(not_stop).item() > 0 # not stop
        return (not not_stop)

    def _stop_cond_grid(self, samples):
        # # print(f"in stop condition, samples: {samples}")
        # equal_to_zero = (samples == 0).long() # equal to zero

        # accum_equal_zero = torch.sum(equal_to_zero.sum(-1), dim=-1)
        # not_stop = (accum_equal_zero == 0).long() # 
        # not_stop = torch.sum(not_stop).item() > 0 # not stop

        # print(f"in stop condition, samples: {samples}")
        equal_to_zero = (samples == 0).long() # equal to zero

        accum_equal_zero = torch.sum(equal_to_zero, dim=-1) ##### number of xyzs equal to 0
        # not_stop = (accum_equal_zero == 0).long() # 
        # not_stop = (accum_equal_zero <= 2).long() # not stop
        stop_indicator = (accum_equal_zero == 3).long() # all three euqalt to zero --> stop
        stop_indicator = torch.any(stop_indicator, dim=-1).long() # bsz; for each batch size
        not_stop = torch.sum(1 - stop_indicator).item() > 0
        # not_stop = torch.sum(not_stop).item() > 0 # not stop
        return (not not_stop)

    def _sample_grids(self, num_samples, context_feature, seq_context, temperature, top_k, top_p, cond_context_info=False, sample_context=None, sampling_max_num_grids=-1):
      loop_idx = 0
      sampling_max_num_grids = sampling_max_num_grids if sampling_max_num_grids > 0 else self.max_num_grids
      grid_xyzs = torch.zeros([num_samples, 1, 3], dtype=torch.long).cuda() # grid_xyzs --> grid_xyzs (num_samples x 1 x 3)
      # grid_content = torch.zeros([num_samples, 1, self.grid_size, self.grid_size, self.grid_size], dtype=torch.long).cuda()
      grid_content = torch.zeros([num_samples, 1], dtype=torch.long).cuda() + (2 ** (self.grid_size ** 3))
      grid_pos = torch.zeros([num_samples, 1], dtype=torch.long).cuda()

    #   grid_xyzs = torch.zeros()
      if cond_context_info:
        ##### grid_xyzs: bsz x (context grids + 1)
        grid_xyzs = torch.cat(
            [sample_context['grid_xyzs'], grid_xyzs], dim=1
        )
        ##### grid_content: bsz x (context grids + 1)
        grid_content = torch.cat(
            [sample_context['grid_content_vocab'], grid_content], dim=1 ##### num_samples x (context_vocab + 1)
        )
        ##### grid_pos: bsz x (context grids + 1)
        grid_pos = torch.cat(
            [sample_context['grid_pos'], sample_context['grid_pos'][:, -1:] + 1], dim=1
        )
        print(f"Condon context info! grid_xyzs: {grid_xyzs.size()}, grid_content: {grid_content.size()}, grid_pos: {grid_pos.size()}")
      samples = {}
      samples['grid_xyzs'] = grid_xyzs
      samples['grid_content_vocab'] = grid_content
      samples['grid_pos'] = grid_pos
      while True:
        # samples 
        loop_idx, samples = self._loop_sample_grid(loop_idx, samples, global_context_embedding=context_feature, sequential_context_embeddings=seq_context, temperature=temperature, top_k=top_k, top_p=top_p)
        
        grid_xyzs = samples['grid_xyzs']; grid_content = samples['grid_content_vocab']; grid_pos = samples['grid_pos']

        if self._stop_cond_grid(grid_xyzs) or loop_idx >= sampling_max_num_grids:
          break
        # grid_xyzs: bsz x (1 + 1 + cur_seq_length) x 3
        grid_xyzs = torch.cat(
          [grid_xyzs, torch.zeros((num_samples, 1, 3), dtype=torch.long).cuda()], dim=1
        )
        # grid_content: bsz x (1 + 1 + cur_seq_length) x (gs x gs x gs)
        grid_content = torch.cat(
          [grid_content, torch.zeros((num_samples, 1), dtype=torch.long).cuda() + (2 ** (self.grid_size ** 3))], dim=1
        )
        grid_pos = torch.cat(
          [grid_pos, torch.full((num_samples, 1), fill_value=grid_pos.size(1), dtype=torch.long).cuda()], dim=1
        )
        samples['grid_xyzs'] = grid_xyzs
        samples['grid_content_vocab'] = grid_content
        samples['grid_pos'] = grid_pos
        # print(f"grid_xyzs: {grid_xyzs.size()}, grid_content: {grid_content.size()}, grid_pos: {grid_pos.size()}")
      samples['grid_xyzs'] = grid_xyzs
      samples['grid_content_vocab'] = grid_content
      samples['grid_pos'] = grid_pos
      return loop_idx, samples
      

    def _sample_vertices(self, num_samples, inputs_embedding_layers, project_to_logits, context_feature, seq_context, temperature, top_k, top_p): # sample
        loop_idx = 0
        # init samples
        samples = torch.zeros([num_samples, 0], dtype=torch.long).cuda()
        probs = torch.ones([num_samples, 0], dtype=torch.float32).cuda()
        while True: # _loop_sample for samples
            # nn_res_samples
            if opt.model.context_window > 0 and opt.model.cut_vertices > 0:
                nn_res_samples = samples.size(1) - opt.model.context_window
                if nn_res_samples >= 1:
                    masked_nn_vertices_samples = nn_res_samples // 3 +  ((nn_res_samples % 3) > 0)
                    cur_samples = samples[:, masked_nn_vertices_samples * 3: ]
                else:
                    cur_samples = samples
            else:
                cur_samples = samples
            # cur_samples
            loop_idx, cur_samples, outputs, cur_sample_probs, next_sample = self._loop_sample(loop_idx, cur_samples, embedding_layers=inputs_embedding_layers, project_to_logits_layer=project_to_logits, context_feature=context_feature, seq_context=seq_context, temperature=temperature, top_k=top_k, top_p=top_p)
            # probs: bsz x n_cur_samples
            samples = torch.cat(
                [samples, next_sample], dim=-1
            )
            probs = torch.cat(
                [probs, cur_sample_probs], dim=1
            )
            if self._stop_cond(samples) or loop_idx >= self.max_sample_length * 3:
                break
        return loop_idx, samples, outputs, probs
    
    def _sample_beam_vertices(self, samples, probs=None, nn_beam=10, max_sample_lenght=None, temperature=1., top_k=0, top_p=0.95, recenter_verts=True, only_return_complete=True, global_context=None, seq_context=None):
        loop_idx = 0
        # if samples.size(1) > 0:
        #     print(f"max_cur_samples: {torch.max(samples)}")
        cur_beam_dist, cur_beam_logits = self._create_dist(
            samples, embedding_layers=self.inputs_embedding_layers, project_to_logits_layer=self.project_to_logits, global_context_embedding=global_context, sequential_context_embeddings=seq_context, top_k=top_k, top_p=top_p, rt_logits=True
        )
        cur_beam_logits = cur_beam_logits[:, -1, :] # beam logits
        # print(f"cur_beam_logits: {cur_beam_logits.size()}")
        # cur_beam_logits = cur_beam_logits.squeeze(1)
        # cur_beam_logits: n_beam x NN --> beam_logits: n_beam x n_beam; beam_logits_idxes: n_beam x n_beam
        beam_logits, beam_logits_idxes = torch.topk(cur_beam_logits, k=nn_beam, dim=-1) # 
        # print(f"beam_logits: {beam_logits}")
        if samples.size(1) == 0:
            cur_samples = beam_logits_idxes[0]
            cur_probs = beam_logits[0]
            return cur_samples.unsqueeze(1), cur_probs

        # probs: n_beam ---> current probabilities
        # print(f"beam_logits: {beam_logits.size()}, probs: {probs.size()}")
        prob_beam_logits = beam_logits * probs.unsqueeze(-1) # n_beam x n_beam --> probabilities
        expanded_prob_beam_logits = prob_beam_logits.contiguous().view(-1).contiguous() # (n_beam x n_beam)
        # print(f"expanded_prob_beam_logits: {expanded_prob_beam_logits.size()}")
        beam_logits, beam_logits_idxes_aft = torch.topk(expanded_prob_beam_logits, k=nn_beam, dim=0) # n_beam
        # print(f"beam_logits: {beam_logits.size()}, beam_logits_idxes_aft: {beam_logits_idxes_aft.size()}")
        beam_logits_row_idxes = beam_logits_idxes_aft // nn_beam
        beam_logits_col_idxes = beam_logits_idxes_aft % nn_beam

        # print(f"beam_logits_idxes: {beam_logits_idxes.size()}, max_beam_logits_row_idxes: {torch.max(beam_logits_row_idxes)},  max_beam_logits_col_idxes: {torch.min(beam_logits_col_idxes)}, max_beam_logits_col_idxes: {torch.max(beam_logits_col_idxes)},  min_beam_logits_col_idxes: {torch.min(beam_logits_col_idxes)}, samples: {samples.size()}")

        selected_beam_logits_idxes = data_utils.batched_index_select(values=beam_logits_idxes, indices=beam_logits_row_idxes, dim=0) # beam_logits_idxes: n_beam x n_beam --> selected beam logits
        # print("1-selected_beam_logits_idxes:", selected_beam_logits_idxes.size())
        # selected_beam_logits_idxes = selected_beam_logits_idxes.contiguous().squeeze(1).contiguous()
        # print("2-selected_beam_logits_idxes:", selected_beam_logits_idxes.size())
        selected_beam_logits_idxes = data_utils.batched_index_select(values=selected_beam_logits_idxes, indices=beam_logits_col_idxes.unsqueeze(1), dim=1)
        selected_beam_logits_idxes = selected_beam_logits_idxes.contiguous().squeeze(1) # 

        # 
        prob_beam_logits = data_utils.batched_index_select(values=prob_beam_logits, indices=beam_logits_row_idxes, dim=0)
        # prob_beam_logits: n_beam
        # print(f"prob_beam_logits: {prob_beam_logits.size()}")
        prob_beam_logits = data_utils.batched_index_select(values=prob_beam_logits, indices=beam_logits_col_idxes.unsqueeze(1), dim=1)
        prob_beam_logits = prob_beam_logits.squeeze(1)
        

        selected_samples = data_utils.batched_index_select(values=samples, indices=beam_logits_row_idxes, dim=0) # n_beam x (samples_len)
        # n_beam x (samples_len + 1)
        # print("selected_samples:", selected_samples.size())
        # selected_samples = selected_samples.squeeze(1)
        selected_samples = torch.cat([selected_samples, selected_beam_logits_idxes.unsqueeze(-1)], dim=-1) # 
        # print(f"selected_samples: {selected_samples.size()}, prob_beam_logits: {prob_beam_logits.size()}")
        return selected_samples, prob_beam_logits
        

    def sample_beam_search_single_sample(self, nn_beam=10, context=None, max_sample_length=None, temperature=1., top_k=0, top_p=0.95, recenter_verts=True, only_return_complete=True):
        # single sample
        # samples
        beam_context = {}
        for k in context: # nn_beam
            cur_beam_context = torch.cat([context[k] for _ in range(nn_beam)], dim=0)
            # cur_beam_context: (nn_beam x other dims)
            beam_context[k] = cur_beam_context # beam context
            # then loop for beam search
        # prepare context...
        global_context, seq_context = self._prepare_context(beam_context)
        loop_idx = 0
        # init samples
        samples = torch.zeros([nn_beam, 0], dtype=torch.long).cuda()
        probs = torch.ones([nn_beam,], dtype=torch.float32).cuda()
        while True: # _loop_sample for samples
            # loop_idx, samples, outputs = self._loop_sample(loop_idx, samples, embedding_layers=self.inputs_embedding_layers, project_to_logits_layer=self.project_to_logits, context_feature=global_context, seq_context=seq_context, top_k=top_k, top_p=top_p)
            
            samples, probs = self._sample_beam_vertices(samples, probs=probs, nn_beam=nn_beam, max_sample_lenght=max_sample_length, temperature=temperature, top_k=top_k, top_p=top_p, recenter_verts=recenter_verts, only_return_complete=only_return_complete, global_context=global_context, seq_context=seq_context)
            if self._stop_cond(samples) or loop_idx >= self.max_sample_length * 3 or samples.size(-1) >= self.max_sample_length * 3:
                break
            loop_idx += 1
        # print("returning...")
        return loop_idx, samples

    # num_of_samples, context, max_sample_length
    def sample_beam_search(self, num_samples, context=None, max_sample_length=None, temperature=1., top_k=0, top_p=0.95, recenter_verts=True, only_return_complete=True):
        # contextual infmation
        # give context from one single instance
        nn_beam = 10
        # global_context, seq_context = self._prepare_context(context)
        tot_vertices = []
        tot_completed = []
        tot_num_vertices = []
        tot_vertices_mask = []
        tot_class_label = []

        for i_s in range(num_samples):
            cur_sample_context = {}
            if context is not None:
                for k in context:
                    cur_context_feature = context[k][i_s: i_s + 1]
                    cur_sample_context[k] = cur_context_feature
            else:
                cur_sample_context = None
            loop_idx, samples = self.sample_beam_search_single_sample(nn_beam=nn_beam, context=cur_sample_context, max_sample_length=max_sample_length, temperature=temperature, top_k=top_k, top_p=top_p, recenter_verts=recenter_verts, only_return_complete=only_return_complete)
            
            expanded_context_label = torch.cat([context['class_label'][i_s: i_s+1] for _ in range(nn_beam)], dim=0)
            
            completed = torch.any(samples == 0, dim=-1) 
            stop_index_completed = torch.argmax((samples == 0).long(), dim=-1)
            stop_index_incomplete = (max_sample_length * 3 * torch.ones_like(stop_index_completed))
            # select the stop indexes
            stop_index = torch.where(completed, stop_index_completed, stop_index_incomplete) # stop index
            num_vertices = torch.floor_divide(stop_index, 3) # 

            tot_completed.append(completed)
            tot_num_vertices.append(num_vertices)
            # tot_vertices_mask.append()
            tot_class_label.append(expanded_context_label)

            print(f"number of vertices: {num_vertices}") # number of vertices for all samples

            v = samples
            v = v[:, :(torch.max(num_vertices) * 3)] - 1
            verts_dequantized = layer_utils.dequantize_verts(v, self.quantization_bits)
            # vertices # 
            vertices = verts_dequantized.contiguous().view(nn_beam, -1, 3).contiguous()
            # z, x, y --> y, x, z?
            vertices = torch.cat([vertices[..., 2].unsqueeze(-1), vertices[..., 1].unsqueeze(-1), vertices[..., 0].unsqueeze(-1)], dim=-1)

            # vertices; 
            if max_sample_length > vertices.size(1):
                pad_size = max_sample_length - vertices.size(1)
                pad_zeros = torch.zeros((nn_beam, pad_size, 3), dtype=torch.float32).cuda()
                vertices = torch.cat([vertices, pad_zeros], dim=1)
            else:
                vertices = vertices[:, :max_sample_length]

            vertices_mask = (torch.arange(0, max_sample_length).unsqueeze(0).cuda() < num_vertices.unsqueeze(1)).float() # valid ones

            tot_vertices_mask.append(vertices_mask)

            ### centralize vertices ### # and use the vertices 
            if recenter_verts:
                vert_max, _ = torch.max( # max pooling for the maximum vertices value
                    vertices - 1e10 * (1. - vertices_mask).unsqueeze(-1), dim=1, keepdim=True)
                vert_min, _ = torch.min( # min pooling for the minimum vertices value
                    vertices + 1e10 * (1. - vertices_mask).unsqueeze(-1), dim=1, keepdim=True)
                vert_centers = 0.5 * (vert_max + vert_min) # centers
                vertices -= vert_centers # centralize vertices # vertices
            vertices *= vertices_mask.unsqueeze(-1)# vertices mask?
            tot_vertices.append(vertices)
        tot_vertices = torch.cat(tot_vertices, dim=0)
        tot_completed = torch.cat(tot_completed, dim=0)
        tot_num_vertices = torch.cat(tot_num_vertices, dim=0)
        tot_vertices_mask = torch.cat(tot_vertices_mask, dim=0)
        tot_class_label = torch.cat(tot_class_label, dim=0)
        outputs = {
            'completed': tot_completed,  #
            'vertices': tot_vertices,  # dequantized vertices
            'num_vertices': tot_num_vertices,
            'vertices_mask': tot_vertices_mask,
            'class_label': tot_class_label
        }
        return outputs # return outputs # 


    def sample(self, num_samples, context=None, max_sample_length=None, temperature=1., top_k=0, top_p=0.95, recenter_verts=True, only_return_complete=True, cond_context_info=False, sampling_max_num_grids=-1): # only the largets value is considered?
        # sample
        global_context, seq_context = self._prepare_context(context)
        # global_contex
        # left_cond, rgt_cond = self._prepare_prediction(global_context)

        # samples = tf.zeros([num_samples, 0], dtype=tf.int32)
        # samples = torch.zeros([num_samples, 0], dtype=torch.long)
        max_sample_length = max_sample_length
        self.max_sample_length = max_sample_length

        # sample vertice;  ######## 
        # loop_idx, samples, outputs, probs = self._sample_vertices(num_samples, inputs_embedding_layers=self.inputs_embedding_layers, project_to_logits=self.project_to_logits, context_feature=global_context, seq_context=seq_context, temperature=temperature, top_k=top_k, top_p=top_p)
        
        # max_samp
        cond_context = None if not cond_context_info else context
        loop_idx, samples = self._sample_grids(num_samples, global_context, seq_context, temperature, top_k, top_p, cond_context_info=cond_context_info, sample_context=cond_context, sampling_max_num_grids=sampling_max_num_grids)
        grid_xyzs = samples['grid_xyzs'] - 1
        grid_values = samples['grid_content_vocab']

        outputs = {
          'grid_xyzs': grid_xyzs,
          'grid_values': grid_values
        }

        # # print(f"ori sampled vertex size: {samples.size()}"
        # # )
        # completed = torch.any(grid_xyzs == 0, dim=-1) 
        # completed = torch.any(completed, dim=-1)
        # stop_index_completed = torch.argmax(((grid_xyzs == 0).long().sum(-1) > 0).long(), dim=-1)
        # stop_index_incomplete = (self.max_num_grids * torch.ones_like(stop_index_completed))
        # # select the stop indexes
        # stop_index = torch.where(completed, stop_index_completed, stop_index_incomplete) # stop index
        # # num_vertices = torch.floor_divide(stop_index, 3) # 

        # # print(f"number of vertices: {num_vertices}") # number of vertices for all samples

        # # print(f"vertex samples size: {samples.size()}")

        # v = grid_xyzs
        # v = v[:, :(torch.max(stop_index))] - 1
        # # probs = probs[:,  :(torch.max(num_vertices) * 3)]
        # verts_dequantized = layer_utils.dequantize_verts(v, self.quantization_bits)
        # # vertices
        # vertices = verts_dequantized.contiguous().view(num_samples, -1, 3).contiguous()
        # # z, x, y --> y, x, z?
        # vertices = torch.cat([vertices[..., 2].unsqueeze(-1), vertices[..., 1].unsqueeze(-1), vertices[..., 0].unsqueeze(-1)], dim=-1)

        # probs = probs.contiguous().view(num_samples, -1, 3).contiguous()
        # probs = torch.cat([probs[..., 2].unsqueeze(-1), probs[..., 1].unsqueeze(-1), probs[..., 0].unsqueeze(-1)], dim=-1)

        # # vertices; 
        # if max_sample_length > vertices.size(1):
        #     pad_size = max_sample_length - vertices.size(1)
        #     pad_zeros = torch.zeros((num_samples, pad_size, 3), dtype=torch.float32).cuda()
        #     # padding
        #     vertices = torch.cat([vertices, pad_zeros], dim=1)
        #     probs = torch.cat([probs, pad_zeros.clone()], dim=1)
        # else:
        #     vertices = vertices[:, :max_sample_length]
        #     probs = probs[:, :max_sample_length]

        # vertices_mask = (torch.arange(0, max_sample_length).unsqueeze(0).cuda() < num_vertices.unsqueeze(1)).float() # valid ones

        # ### centralize vertices ### # and use the vertices 
        # if recenter_verts:
        #     vert_max, _ = torch.max( # max pooling for the maximum vertices value
        #         vertices - 1e10 * (1. - vertices_mask).unsqueeze(-1), dim=1, keepdim=True)
        #     vert_min, _ = torch.min( # min pooling for the minimum vertices value
        #         vertices + 1e10 * (1. - vertices_mask).unsqueeze(-1), dim=1, keepdim=True)
        #     vert_centers = 0.5 * (vert_max + vert_min) # centers
        #     vertices -= vert_centers # centralize vertices # vertices
        # vertices *= vertices_mask.unsqueeze(-1)# vertices mask? # vertices_mask: bsz x max_sample_length
        # # vertices_mask_flat = vertices_mask.unsqueeze(-1).contiguous().repeat(1, 1, 3).contiguous()
        # # vertices_mask_flat = vertices_mask_flat.contiguous().view(vertices_mask_flat.size(0), -1).contiguous()
        # # probs *= vertices_mask_flat # .unsqueeze(-1) # probs
        # probs *= vertices_mask.unsqueeze(-1)

        # outputs = {
        #     'completed': completed,  #
        #     'vertices': vertices,  # dequantized vertices
        #     'num_vertices': num_vertices,
        #     'vertices_mask': vertices_mask,
        #     'class_label': context['class_label'],
        #     'probs': probs
        # }

        return outputs


