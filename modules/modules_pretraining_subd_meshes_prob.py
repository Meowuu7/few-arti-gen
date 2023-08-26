# import sonnet as snt
# from tensor2tensor.layers import common_attention
# from tensor2tensor.layers import common_layers
# import tensorflow.compat.v1 as tf
# from tensorflow.python.framework import function
# import tensorflow_probability as tfp

from cmath import log
from glob import glob
from tracemalloc import is_tracing
from turtle import reset
import numpy as np
import torch.nn as nn
import layer_utils
import torch
from utils.data_utils_torch import batched_index_select
import utils.data_utils_torch as data_utils
import math

from options.options import opt
from utils.dataset_utils import read_edges, upsample_vertices



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

    def forward(self, inputs, sequential_context_embeddings=None, mask=None, sequential_context_mask=None, context_window=None, prefix_length=None):
        # inputs: total inputs; sequential_context_embeddings: sequential context embedding
        ### 
        context_window = opt.model.context_window if context_window is None else context_window
        seq_length = inputs.size(1)
        # bsz = inputs.size(0) # inputs
        
        # seq_length x seq_length --> attentiion
        atten_mask = np.tri(seq_length, seq_length, -1.0, dtype=np.float32).T # tri
        # atten_mask = np.tri(seq_length, seq_length, 0.0, dtype=np.float32)
        atten_mask = torch.from_numpy(atten_mask).float().cuda()
        self.atten_mask = atten_mask

        # if context_window > 0: # context window # a positive context dinwod
        #     context_atten_mask = np.tri(seq_length, seq_length, -1.0 * float(context_window), dtype=np.float32)
        #     context_atten_mask = torch.from_numpy(context_atten_mask).float().cuda()
        #     atten_mask = context_atten_mask + atten_mask

        prefix_length = self.prefix_key_len if prefix_length is None else prefix_length
        
        # context attention mask
        atten_mask = (atten_mask > 0.5)
        atten_mask[:, :prefix_length] = False
        
        if sequential_context_embeddings is not None:
            sequential_context_embeddings = sequential_context_embeddings[:, 1:]
            seq_atten_mask = atten_mask.clone()
            seq_atten_mask = seq_atten_mask[:, :-1]

        atten_mask = None

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
        if self.layer_norm: #### layer_norm ####
            x = self.out_layer_norm(x)
        # not self embedding
        # x = x[:, :-1]
        # x = x[:, self.prefix_key_len - 1: -1]
        if atten_mask is None:
            x = x[:, prefix_length: ]
        else:
            x = x[:, prefix_length - 1: -1]
        # x = x[:, self.prefix_key_len: ]
        return x


#### prompt finetuning ####
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
        self.num_grids_quantization = self.num_grids_quantization  #  * 2
        self.num_grids_quantization = 2 ** self.quantization_bits
        
        self.vocab_size = 2 ** (self.grid_size ** 3) + 5
        print(f"Constructing VertexModel with vocab size: {self.vocab_size}, grid size: {self.grid_size}, num_grids_quantization: {self.num_grids_quantization}.")

        self.grid_pos_embed_max_num = self.num_grids_quantization ** 3
        self.grid_pos_embed_max_num = self.max_num_grids
        if opt.dataset.use_context_window_as_max:
            self.grid_pos_embed_max_num = opt.model.context_window
            self.max_num_grids = opt.model.context_window
        print(f"Maximum grid position embedding number: {self.grid_pos_embed_max_num}.")
        self.prefix_key_len = opt.model.prefix_key_len
        self.prefix_value_len = opt.model.prefix_value_len

        self.use_multi_gpu = use_multi_gpu
        self.num_parts = opt.dataset.num_parts
        self.ar_object = opt.dataset.ar_object
        self.num_objects = opt.dataset.num_objects

        self.st_subd_idx = opt.dataset.st_subd_idx

        self.multi_part_not_ar_object = (self.num_parts > 1) and (not self.ar_object)
        self.use_local_frame = opt.dataset.use_local_frame
        
        subdn = opt.dataset.subdn 
        print(f"Subdn: {subdn}")

        #### discrete position encoding ####
        # ### for grid content encoding ### # grid content encoding #
        # self.grid_embedding_layers = nn.ModuleList(
        #   [
        #     nn.Embedding(num_embeddings=self.grid_pos_embed_max_num + 5, embedding_dim=self.embedding_dim), # grid order embedding
        #     nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim), # for grid xyz's embedding; xyz discrete position encodings
        #     # nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim), # for grid xyz's embedding;
        #     # nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim),
        #     nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim), # xyz indicator embedding
        #   ]
        # )
        #### discrete position encoding ####

        # ### for grid content encoding ### # grid content encoding #
        # self.grid_prefix_embedding_layers = nn.ModuleList(
        #   [
        #     nn.Embedding(num_embeddings=self.grid_pos_embed_max_num + 5, embedding_dim=self.embedding_dim), # grid order embedding
        #     nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim), # for grid xyz's embedding; xyz discrete position encodings
        #     nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim), # xyz indicator embedding
        #   ]
        # )

        # self.subd_to_grid_embedding_layers = nn.ModuleDict( {
        #     str(i_subd): nn.ModuleList(
        #         [
        #             nn.Embedding(num_embeddings=self.grid_pos_embed_max_num + 5, embedding_dim=self.embedding_dim), # grid order embedding
        #             nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim), # for grid xyz's embedding; xyz discrete position encodings
        #             nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim), # xyz indicator embedding
        #         ]
        #     ).cuda() for i_subd in range(subdn - 1)
        # }).cuda()
        # # self.subd_to_grid_embedding_layers = nn.ModuleList(subd_to_grid_embedding_layers)
        
        # self.subd_to_grid_prefix_embedding_layers = [
        #     nn.ModuleList(
        #         [
        #             nn.Embedding(num_embeddings=self.grid_pos_embed_max_num + 5, embedding_dim=self.embedding_dim), # grid order embedding
        #             nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim), # for grid xyz's embedding; xyz discrete position encodings
        #             nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim), # xyz indicator embedding
        #         ]
        #     ).cuda() for i_subd in range(subdn - 1)
        # ]
        # self.subd_to_grid_prefix_embedding_layers = nn.ModuleList(self.subd_to_grid_prefix_embedding_layers).cuda()


        # self.subd_to_grid_embedding_layers = nn.ModuleDict( {
        #     str(i_subd): nn.ModuleList(
        #         [
        #             nn.Embedding(num_embeddings=self.grid_pos_embed_max_num + 5, embedding_dim=self.embedding_dim),
        #             nn.Sequential(
        #                 nn.Linear(in_features=3, out_features=self.embedding_dim, bias=True),
        #                 nn.ReLU(),
        #                 nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim * 2, bias=True),
        #                 nn.ReLU(),
        #                 nn.Linear(in_features=self.embedding_dim * 2, out_features=self.embedding_dim * 3, bias=True),
        #             ),
        #             nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim), # xyz indicator embedding
        #             # nn.ReLU(),

        #             # nn.Embedding(num_embeddings=self.grid_pos_embed_max_num + 5, embedding_dim=self.embedding_dim), # grid order embedding
        #             # nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim), # for grid xyz's embedding; xyz discrete position encodings
        #             # nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim), # xyz indicator embedding
        #         ]
        #     ).cuda() for i_subd in range(subdn - 1)
        # }).cuda()

        self.subd_to_grid_embedding_layers = nn.ModuleDict( {
            str(i_subd): nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=self.grid_pos_embed_max_num + 5, embedding_dim=self.embedding_dim),
                    nn.Sequential(
                        nn.Linear(in_features=3, out_features=self.embedding_dim, bias=True),
                        nn.ReLU(),
                        nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim * 2, bias=True),
                        nn.ReLU(),
                        nn.Linear(in_features=self.embedding_dim * 2, out_features=self.embedding_dim * 3, bias=True),
                    ),
                    nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim), # xyz indicator embedding
                    # nn.ReLU(),

                    # nn.Embedding(num_embeddings=self.grid_pos_embed_max_num + 5, embedding_dim=self.embedding_dim), # grid order embedding
                    # nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim), # for grid xyz's embedding; xyz discrete position encodings
                    # nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim), # xyz indicator embedding
                ]
            ).cuda() for i_subd in range(subdn - 1)
        }).cuda()
        # self.subd_to_grid_embedding_layers = nn.ModuleList(subd_to_grid_embedding_layers)
        
        self.subd_to_grid_prefix_embedding_layers = [
            nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=self.grid_pos_embed_max_num + 5, embedding_dim=self.embedding_dim),
                    nn.Sequential(
                        nn.Linear(in_features=3, out_features=self.embedding_dim, bias=True),
                        nn.ReLU(),
                        nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim * 2, bias=True),
                        nn.ReLU(),
                        nn.Linear(in_features=self.embedding_dim * 2, out_features=self.embedding_dim * 3, bias=True),
                    ),
                    nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim), # xyz indicator embedding
                    # nn.ReLU(),
                    # nn.Embedding(num_embeddings=self.grid_pos_embed_max_num + 5, embedding_dim=self.embedding_dim), # grid order embedding
                    # nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim), # for grid xyz's embedding; xyz discrete position encodings
                    # nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim), # xyz indicator embedding
                ]
            ).cuda() for i_subd in range(subdn - 1)
        ]
        self.subd_to_grid_prefix_embedding_layers = nn.ModuleList(self.subd_to_grid_prefix_embedding_layers).cuda()

        self.subd_to_half_flap_embedding_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(in_features=self.embedding_dim * 4, out_features=self.embedding_dim * 2, bias=True),
                nn.ReLU(),
                nn.Linear(in_features=self.embedding_dim * 2, out_features=self.embedding_dim * 2, bias=True),
                nn.ReLU(),
                nn.Linear(in_features=self.embedding_dim * 2, out_features=self.embedding_dim, bias=True),
            ) for i_subd in range(subdn - 1)]
        )
        self.subd_to_half_flap_embedding_layers = self.subd_to_half_flap_embedding_layers.cuda()

        ##### context pts embeddingss #####
        self.context_subd_to_grid_embedding_layers = nn.ModuleDict( {
            str(i_subd): nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=self.grid_pos_embed_max_num + 5, embedding_dim=self.embedding_dim),
                    nn.Sequential(
                        nn.Linear(in_features=3, out_features=self.embedding_dim, bias=True),
                        nn.ReLU(),
                        nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim * 2, bias=True),
                        nn.ReLU(),
                        nn.Linear(in_features=self.embedding_dim * 2, out_features=self.embedding_dim * 3, bias=True),
                    ),
                    nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim), # xyz indicator embedding
                    # nn.ReLU(),

                    # nn.Embedding(num_embeddings=self.grid_pos_embed_max_num + 5, embedding_dim=self.embedding_dim), # grid order embedding
                    # nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim), # for grid xyz's embedding; xyz discrete position encodings
                    # nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim), # xyz indicator embedding
                ]
            ).cuda() for i_subd in range(subdn - 1)
        }).cuda()
        # self.subd_to_grid_embedding_layers = nn.ModuleList(subd_to_grid_embedding_layers)

        self.context_local_frame_embedding_layers = [
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(in_features=3, out_features=self.embedding_dim, bias=True),
                        nn.ReLU(),
                        nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim * 2, bias=True),
                        nn.ReLU(),
                        nn.Linear(in_features=self.embedding_dim * 2, out_features=self.embedding_dim * 3, bias=True),
                    ),
                    nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim)
                ]
            ).cuda() for i_subd in range(subdn - 1)
        ]
        self.context_local_frame_embedding_layers = nn.ModuleList(self.context_local_frame_embedding_layers).cuda()
        
        ##### context grid embeddings #####
        self.context_subd_to_grid_prefix_embedding_layers = [
            nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=self.grid_pos_embed_max_num + 5, embedding_dim=self.embedding_dim),
                    nn.Sequential(
                        nn.Linear(in_features=3, out_features=self.embedding_dim, bias=True),
                        nn.ReLU(),
                        nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim * 2, bias=True),
                        nn.ReLU(),
                        nn.Linear(in_features=self.embedding_dim * 2, out_features=self.embedding_dim * 3, bias=True),
                    ),
                    nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim), # xyz indicator embedding
                    # nn.ReLU(),
                    # nn.Embedding(num_embeddings=self.grid_pos_embed_max_num + 5, embedding_dim=self.embedding_dim), # grid order embedding
                    # nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim), # for grid xyz's embedding; xyz discrete position encodings
                    # nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim), # xyz indicator embedding
                ]
            ).cuda() for i_subd in range(subdn - 1)
        ]
        self.context_subd_to_grid_prefix_embedding_layers = nn.ModuleList(self.context_subd_to_grid_prefix_embedding_layers).cuda()

        # self.grid_content_embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)

        # # grid embedding layers; treat them as tokens
        # self.grid_coord_embedding_layers = nn.ModuleList(
        #   [
        #     nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim), # for grid coordinate embedding
        #     nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim),
        #     nn.Embedding(num_embeddings=self.num_grids_quantization + 5, embedding_dim=self.embedding_dim),
        #   ]
        # )

        # context and content embedding layers #
        self.subd_to_content_context_embedding_layers = nn.ModuleList(
            [ 
                nn.Embedding(num_embeddings=2, embedding_dim=self.embedding_dim) for _ in range(subdn - 1)
            ]
        )

        # decod
        decoder_config['with_seq_context'] = False
        self.subd_to_decoder_grid_coord = {}
        subdn = opt.dataset.subdn
        for i_subd in range(subdn - 1):
            
            if not self.use_local_frame:
                self.subd_to_decoder_grid_coord[i_subd] = TransformerDecoderGrid(**decoder_config)
            else:        
                #### MLP for decoding
                self.subd_to_decoder_grid_coord[i_subd] = nn.Sequential(
                            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim * 2, bias=True),
                            nn.ReLU(),
                            nn.Linear(in_features=self.embedding_dim * 2, out_features=self.embedding_dim * 2, bias=True),
                            nn.ReLU(),
                            nn.Linear(in_features=self.embedding_dim * 2, out_features=self.embedding_dim, bias=True),
                        )

        # decoder_config['with_seq_context'] = False
        # self.decoder_grid_coord = TransformerDecoderGrid(**decoder_config)
        
        # self.decoder_inner_grid_content = TransformerDecoderInnerGrid(**decoder_config)
        # project to logits, grid coordinates
        self.subd_to_grid_project_to_logits = [
            nn.ModuleList( # 
                [
                    nn.Linear(self.embedding_dim, self.num_grids_quantization + 1, bias=True),
                    nn.Linear(self.embedding_dim, self.num_grids_quantization + 1, bias=True),
                    nn.Linear(self.embedding_dim, self.num_grids_quantization + 1, bias=True),
                ]
            ).cuda() for i_subd in range(subdn - 1)
        ]
        self.subd_to_grid_project_to_logits = nn.ModuleList(self.subd_to_grid_project_to_logits).cuda()
        # print(len(self.subd_to_grid_project_to_logits), type(self.sub))
        
        
        # self.grid_project_to_logits = nn.ModuleList(
        #   [
        #     nn.Linear(self.embedding_dim, self.num_grids_quantization + 1, bias=True),
        #     nn.Linear(self.embedding_dim, self.num_grids_quantization + 1, bias=True),
        #     nn.Linear(self.embedding_dim, self.num_grids_quantization + 1, bias=True),
        #   ]
        # )
        
        # decode grid content from embeded latent vectors
        # self.grid_content_project_to_logits = nn.ModuleList(
        #   [nn.Linear(self.embedding_dim, 2, bias=True) for _ in range(self.grid_size ** 3)]
        # )

        ##### content #####
        # self.grid_content_project_to_logits = nn.Linear(self.embedding_dim, self.vocab_size, bias=True) # project to logits for grid content decoding


        self.class_embedding_layer = nn.Parameter(
            torch.zeros((self.num_classes, self.prefix_key_len, self.embedding_dim), requires_grad=True, dtype=torch.float32), requires_grad=True
        )

        if not self.class_conditional: # class condition # claass condition
            self.zero_embed = nn.Parameter(torch.zeros(size=(1, 1, self.embedding_dim), requires_grad=True, dtype=torch.float32), requires_grad=True)


    def _prepare_context(self, context, adapter_module=None):
      # - content embedding for each grid --> [gird content, grid coord embeddings, grid order embeddings]
      # - grid coord_order embedding --> [grid coord embeddings, grid order embeddings]
        if self.class_conditional: ### class conditional
            bsz = context['class_label'].size(0)
            if adapter_module is None:
                global_context_embedding = self.class_embedding_layer[context['class_label']] # class_label: bsz x 1 --> embedding: bsz x 1 x key_len x embedding_dim
                global_context_embedding = global_context_embedding.squeeze(1)
                # global_context_embedding_content = self.class_embedding_layer_content[context['class_label']]
                # global_context_embedding_grid = self.class_embedding_layer_grid[context['class_label']] # class_label: bsz x 1 --> embedding: bsz x 1 x key_len x embedding_dim
                # # print("embeddings", global_context_embedding.size(), global_context_embedding_content.size(), global_context_embedding_grid.size())
                # # #### global_context_embedding ####
                # # global_context_embedding = global_context_embedding.squeeze(1)
                # global_context_embedding_content = global_context_embedding_content.squeeze(1)
                # global_context_embedding_grid = global_context_embedding_grid.squeeze(1)
            else:
                #### class_embedding_layer: 1 x prefix_length x embedding_dim ####
                global_context_embedding = adapter_module.prompt_values[0] # class_label: bsz x 1 --> embedding: bsz x 1 x key_len x embedding_dim
                # global_context_embedding_content = adapter_module.prompt_values[1]
                # global_context_embedding_grid = adapter_module.prompt_values[2] # class_label: bsz x 1 --> embedding: bsz x 1 x key_len x embedding_dim
                # print("embeddings", global_context_embedding.size(), global_context_embedding_content.size(), global_context_embedding_grid.size())
                # #### global_context_embedding ####
                global_context_embedding = global_context_embedding.contiguous().repeat(bsz, 1, 1).contiguous() # bsz x prefix_length x embedding_dim
                # global_context_embedding_content = global_context_embedding_content.contiguous().repeat(bsz, 1, 1).contiguous() # bsz x prefix_length x embedding_dim
                # global_context_embedding_grid = global_context_embedding_grid.contiguous().repeat(bsz, 1, 1).contiguous() # bsz x prefix_length x embedding_dim
            
            # global_context_embedding = [global_context_embedding, global_context_embedding_content, global_context_embedding_grid]
            global_context_embedding = global_context_embedding
        else:
            global_context_embedding = None
        return  global_context_embedding, None


    def _embed_input_grids(self, grid_xyzs, grid_content, grid_pos=None, global_context_embedding=None):
      # grid content: bsz x grid_length --> should convert grids into discrete grid content values in the input
      # grid xyzs: bsz x grid_length x 3
      global_context_embedding, global_context_embedding_content, global_context_embedding_grid = global_context_embedding
    #   print(f"Max of grid_xyzs: {torch.max(grid_xyzs)}, min of grid_xyzs: {torch.min(grid_xyzs)}, max of grid_content: {torch.max(grid_content)}, min of grid_content: {torch.min(grid_content)}, max of grid_pos: {torch.max(grid_pos)}, min of grid_pos: {torch.min(grid_pos)}")
    #   print(f"grid_xyzs: {grid_xyzs.size()}, grid_content: {grid_content.size()}, grid_pos: {grid_pos.size()}")
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

    #   print("Here 1")
      # 
    #   grid_coord_embeddings = []
      for c in range(nn_grid_xyzs):
        cur_grid_coord = grid_xyzs[:, :, c] # bsz x grid_length
        # cur_grid_coord_embedding: bsz x grid_length x embedding_dim
        cur_grid_coord_embedding = self.grid_embedding_layers[c + 1](cur_grid_coord) 
        grid_coord_embedding += cur_grid_coord_embedding # cur_grid_coord_embedding: bsz x grid_
        # grid_coord_embeddings.append(cur_grid_coord_embedding.unsqueeze(2))
      
    #   print("Here 2")
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
      

      grid_content_embedding = self.grid_content_embedding_layer(grid_content) # grid_content: bsz x grid_length --> grid_content_embedding: bsz x grid_length x embedding_size
      grid_content_embedding = grid_content_embedding + grid_embedding # grid_embedding: bsz x grid_length x embedding_size
      
    #   print("Here 4")

      grid_xyz = torch.arange(0, 3, dtype=torch.long).cuda()
      grid_xyz_embeddings = self.grid_embedding_layers[-1](grid_xyz) # 3 x embedding_dim
      grid_xyz_embeddings = grid_xyz_embeddings.unsqueeze(0).unsqueeze(0).contiguous().repeat(bsz, grid_length, 1, 1).contiguous() # order_embedding: bsz x 
      # grid order embedding and grid xyz embedding?
      # grid xyz embedding

    #   print("Here 5")
      
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

    def _embed_pts(self, cur_subd_prefix_pts, embedding_layers, context_content_indicator_embedding_layers, context_content_indicator=0):
        # print(f"In embed pts, cur_subd_prefix_pts: {cur_subd_prefix_pts.size()}, context_content_indicator: {context_content_indicator}")
        n_verts = cur_subd_prefix_pts.size(1)
        # n_verts = cur_subd_upsample.size(1)
        ##### verts_order #####
        verts_order = torch.arange(n_verts).cuda()
        ##### verts order embedding #####
        verts_order_embedding = embedding_layers[0](verts_order).unsqueeze(0) # bsz x n_verts x embedding_dim
        ##### xyz indicator #####
        xyz_indicator = torch.arange(3).cuda()
        ##### xyz_indicator_embedding #####
        verts_xyz_indicator_embedding = embedding_layers[-1](xyz_indicator).unsqueeze(0) # bsz x 3 x embedding_dim

        ##### 
        # cur_subd_prefix_pts_np = cur_subd_prefix_pts.detach().cpu().numpy()
        # dequan_cur_subd_prefix_pts_np = data_utils.dequantize_verts(cur_subd_prefix_pts_np, n_bits=self.quantization_bits)
        # cur_subd_prefix_pts = torch.from_numpy(dequan_cur_subd_prefix_pts_np).float().cuda()

        cur_subd_upsample_vert_embedding = embedding_layers[1](cur_subd_prefix_pts) # bsz x n_verts x 3 x embedding_dim

        cur_subd_upsample_vert_embedding = cur_subd_upsample_vert_embedding.contiguous().view(cur_subd_upsample_vert_embedding.size(0), cur_subd_upsample_vert_embedding.size(1), 3, -1).contiguous()

        # nex_subd_gt_vert_embedding = self.grid_embedding_layers[2](nex_subd_gt) # bsz x n_verts x 3 x embedding_dim
        # print(f"vert embedding: {cur_subd_upsample_vert_embedding.size()}, xyz_indicator: {verts_xyz_indicator_embedding.size()}, order embedding: {verts_order_embedding.size()}")
        cur_subd_upsample_vert_embedding = cur_subd_upsample_vert_embedding + verts_order_embedding.unsqueeze(2) + verts_xyz_indicator_embedding.unsqueeze(1) # bsz x n_verts x 3 x embedding_dim


        bsz = cur_subd_prefix_pts.size(0)
        n_verts = cur_subd_prefix_pts.size(1)

        ## bsz x n_verts
        cur_subd_content_embedding_indicators = torch.full((bsz, n_verts), fill_value=context_content_indicator, dtype=torch.long).cuda()

        cur_subd_content_embedding = context_content_indicator_embedding_layers(cur_subd_content_embedding_indicators) # bsz x n_verts x embedding_dim
        ##### bsz x n_verts x 3 x embedding_dim ##### 
        cur_subd_upsample_vert_embedding = cur_subd_upsample_vert_embedding + cur_subd_content_embedding.unsqueeze(2)

        # bsz x (n_verts x 3) x embedding_dim ####
        ##### bsz x n_verts x 3 x embedding_dim #####
        # cur_subd_upsample_vert_embedding = cur_subd_upsample_vert_embedding.contiguous().view(bsz, n_verts * 3, -1).contiguous()
        return cur_subd_upsample_vert_embedding

    def _embed_local_frame_pts(self, cur_subd_prefix_pts, embedding_layers):
        # print(f"In embed pts, cur_subd_prefix_pts: {cur_subd_prefix_pts.size()}, context_content_indicator: {context_content_indicator}")
        n_verts = cur_subd_prefix_pts.size(1)
        # n_verts = cur_subd_upsample.size(1)
        ##### verts_order #####
        # verts_order = torch.arange(n_verts).cuda()
        ##### verts order embedding #####
        # verts_order_embedding = embedding_layers[0](verts_order).unsqueeze(0) # bsz x n_verts x embedding_dim
        ##### xyz indicator #####
        xyz_indicator = torch.arange(3).cuda()
        ##### xyz_indicator_embedding ##### indicators
        verts_xyz_indicator_embedding = embedding_layers[-1](xyz_indicator).unsqueeze(0) # bsz x 3 x embedding_dim
        verts_xyz_value_embedding = embedding_layers[0](cur_subd_prefix_pts) # bsz x nn_verts x (3 * embedding_dim)
        ##### verts_xyz_value_embedding: bsz x nn_verts x 3 x embedding_dim #####
        verts_xyz_value_embedding = verts_xyz_value_embedding.contiguous().view(verts_xyz_value_embedding.size(0), verts_xyz_value_embedding.size(1), 3, -1).contiguous()
        ##### verts_xyz_value_embedding: bsz x nn_verts x 3 x embedding_dim ##### #### nn_verts x 3 
        verts_xyz_value_embedding = verts_xyz_value_embedding + verts_xyz_indicator_embedding.contiguous().unsqueeze(1).contiguous()

        # ##### 
        # cur_subd_prefix_pts_np = cur_subd_prefix_pts.detach().cpu().numpy()
        # dequan_cur_subd_prefix_pts_np = data_utils.dequantize_verts(cur_subd_prefix_pts_np, n_bits=self.quantization_bits)
        # cur_subd_prefix_pts = torch.from_numpy(dequan_cur_subd_prefix_pts_np).float().cuda()
        # cur_subd_upsample_vert_embedding = embedding_layers[1](cur_subd_prefix_pts) # bsz x n_verts x 3 x embedding_dim

        # cur_subd_upsample_vert_embedding = cur_subd_upsample_vert_embedding.contiguous().view(cur_subd_upsample_vert_embedding.size(0), cur_subd_upsample_vert_embedding.size(1), 3, -1).contiguous()

        # # nex_subd_gt_vert_embedding = self.grid_embedding_layers[2](nex_subd_gt) # bsz x n_verts x 3 x embedding_dim
        # # print(f"vert embedding: {cur_subd_upsample_vert_embedding.size()}, xyz_indicator: {verts_xyz_indicator_embedding.size()}, order embedding: {verts_order_embedding.size()}")
        # cur_subd_upsample_vert_embedding = cur_subd_upsample_vert_embedding + verts_order_embedding.unsqueeze(2) + verts_xyz_indicator_embedding.unsqueeze(1) # bsz x n_verts x 3 x embedding_dim


        # bsz = cur_subd_prefix_pts.size(0)
        # n_verts = cur_subd_prefix_pts.size(1)

        # ## bsz x n_verts
        # cur_subd_content_embedding_indicators = torch.full((bsz, n_verts), fill_value=context_content_indicator, dtype=torch.long).cuda()

        # cur_subd_content_embedding = context_content_indicator_embedding_layers(cur_subd_content_embedding_indicators) # bsz x n_verts x embedding_dim
        # ##### bsz x n_verts x 3 x embedding_dim ##### 
        # cur_subd_upsample_vert_embedding = cur_subd_upsample_vert_embedding + cur_subd_content_embedding.unsqueeze(2)

        # bsz x (n_verts x 3) x embedding_dim ####
        ##### bsz x n_verts x 3 x embedding_dim #####
        # cur_subd_upsample_vert_embedding = cur_subd_upsample_vert_embedding.contiguous().view(bsz, n_verts * 3, -1).contiguous()
        return verts_xyz_value_embedding


    def _embed_pts_with_contextual_information(self, content_embedding, context_embedding, edges):
        # content_embedding: bsz x nn_verts x 3 x embedding_dim; 
        # context_embedding: bsz x nn_verts x 3 x embedding_dim;
        # edges: bsz x 2 x nn_edges_verts ##### bsz x 2 x nn_edges_verts #####
        fr_edges = edges[:, 0, :] # bsz x nn_edges_verts
        ed_edges = edges[:, 1, :]
        #### with contextual information #### 
        fr_edges, ed_edges = fr_edges - 1, ed_edges - 1 

        # fr_edges_embedding = batched_index_select(values=context_embedding, )
        #### context 
        content_context_embedding = torch.zeros_like(context_embedding) # bsz x nn_verts x 3 x embedding_dim
        #### selected: bsz x nn_edges_verts x  #### context embedding...
        # ## context_embedding: bsz x nn_verts x 3 x embedding_dim --> bsz x nn_edges_verts x 3 x dim
        content_context_embedding[:, fr_edges, :] += batched_index_select(values=context_embedding, indices=ed_edges, dim=1)
        content_embedding = content_embedding + (content_context_embedding / 6.)
        return content_embedding

    def _embed_pts_with_contextual_information_local_frame(self, vertices, edges, i_subd):
        # content_embedding: bsz x nn_verts x 3 x embedding_dim; 
        # context_embedding: bsz x nn_verts x 3 x embedding_dim;
        # edges: bsz x 2 x nn_edges_verts ##### bsz x 2 x nn_edges_verts #####
        fr_edges = edges[:, 0, :] # bsz x nn_edges_verts
        ed_edges = edges[:, 1, :]
        #### with contextual information #### 
        fr_edges, ed_edges = fr_edges - 1, ed_edges - 1 

        # fr_edges 
        dequan_verts = data_utils.dequantize_verts_torch(vertices) ### dequanitze vertices: bsz x nn_verts x 3 #### dequantized vertices
        
        ### batch
        ## fr_edges: bsz x nn_edges_verts ## vertices: bsz x nn_verts x 3
        edges_fr_verts = batched_index_select(values=dequan_verts, indices=fr_edges, dim=1) ### bsz x nn_edges_verts x 3
        edges_to_verts = batched_index_select(values=dequan_verts, indices=ed_edges, dim=1) ### bsz x nn_edges_verts x 3
        diff_edges_fr_to_verts = edges_to_verts - edges_fr_verts ### local frame vertices offset ### bsz x nn_edges_verts x 3
        # diff_edges_fr_to_verts_embeddings: bsz x nn_edges_verts x 3 x embedding_dim
        diff_edges_fr_to_verts_embeddings = self._embed_local_frame_pts(diff_edges_fr_to_verts, self.context_local_frame_embedding_layers[i_subd])
        
        # 

        # fr_edges_embedding = batched_index_select(values=context_embedding, )
        #### context 
        # content_context_embedding = torch.zeros_like(diff_edges_fr_to_verts_embeddings) # bsz x nn_verts x 3 x embedding_dim
        content_context_embedding = torch.zeros((dequan_verts.size(0), dequan_verts.size(1), 3, self.embedding_dim), dtype=torch.float32).cuda()
        #### selected: bsz x nn_edges_verts x  #### context embedding...
        # ## context_embedding: bsz x nn_verts x 3 x embedding_dim --> bsz x nn_edges_verts x 3 x dim
        content_context_embedding[:, fr_edges, :] += diff_edges_fr_to_verts_embeddings
        # content_embedding = content_embedding + (content_context_embedding / 6.)
        content_embedding = content_context_embedding
        return content_embedding


    def _embed_input(self, batch, i_subd=None, global_context_embedding=None):
      # grid content: bsz x grid_length --> should convert grids into discrete grid content values in the input
      # grid xyzs: bsz x grid_length x 3
      # global_context_embedding,  = global_context_embedding

      subd_to_verts_embedding = {}
      # grid_xyzs: n_grids x 3
      subdn = opt.dataset.subdn # if self.training else opt.dataset.subdn_test

      
      i_subd_for_embedding = range(self.st_subd_idx, subdn - 1) if i_subd is None else [i_subd] #### 

      sampled_subd_to_pts = {}
      
    #   print(f"i_subd_for_embedding: {i_subd_for_embedding}")
      for i_subd in i_subd_for_embedding: ###### subdn_to_embeddings
        # if not self.training:
        #     print("here...4")

        # print(f"processing i_subd: {i_subd}")

        if f'subd_{i_subd}' not in batch:
            continue
        # if not self.training:
        #     print("here...5")
        #### cur_subd_upsample ####
        cur_subd_upsample = batch[f'subd_{i_subd}_upsample'] # bsz x n_verts x 3
        # cur_subd_upsample = batch[f'subd_{i_subd + 1}_gt'] # bsz x n_verts x 3
        # nex_subd_gt = batch[f'subd_{i_subd + 1}_gt'] # bsz x n_verts x 3
        cur_subd_prefix_pts = batch[f'subd_{i_subd}']
        cur_subd_edges = batch[f'subd_{i_subd}_edges'] # bsz x 2 x nn_edges_verts 
        nex_subd_edges = batch[f'subd_{i_subd}_upsample_edges']

        # if self.training:
        #         print(f"subd_prefix_pts: {cur_subd_prefix_pts.size()}, max_subd_prefix_pts: {cur_subd_prefix_pts.max()}, min_subd_prefix_pts: {cur_subd_prefix_pts.min()}")
        

        bsz, nn_verts = cur_subd_upsample.size(0), cur_subd_upsample.size(1)
        nn_subd_prefix_pts = cur_subd_prefix_pts.size(1)
        # cur_subd_prefix_embedding = self._embed_pts(cur_subd_prefix_pts, self.grid_prefix_embedding_layers)
        # cur_subd_upsample_vert_embedding = self._embed_pts(cur_subd_upsample, self.grid_embedding_layers)

        # print(f"cur_subd_prefix_pts : {cur_subd_prefix_pts.size()}, cur_subd_upsample: {cur_subd_upsample.size()}, max_cur_subd_upsample: {cur_subd_upsample.max()}, min_cur_subd_upsample: {cur_subd_upsample.min()}")


        # if cur_subd_prefix_pts.size(1) == 0:
        #     # if not self.training:
        #     #     print("here!")
        #     if self.training:
        #         print(f"here with prefix pts size(1) == 0 !")
        #     cur_subd_prefix_embedding = torch.zeros([cur_subd_prefix_pts.size(0), 0, self.embedding_dim], dtype=torch.float32).cuda()
        # else:  
        #     # if not self.training:
        #     #     print("here 3!")
        #     #### embed_pts ####
        #     #### 
        #     if self.training:
        #         print(f"subd_prefix_pts: {cur_subd_prefix_pts.size()}, max_subd_prefix_pts: {cur_subd_prefix_pts.max()}, min_subd_prefix_pts: {cur_subd_prefix_pts.min()}")
        #     cur_subd_prefix_embedding = self._embed_pts(cur_subd_prefix_pts, self.subd_to_grid_prefix_embedding_layers[i_subd], self.subd_to_content_context_embedding_layers[i_subd], context_content_indicator=0)

        #     context_cur_subd_prefix_embedding = self._embed_pts(cur_subd_prefix_pts, self.context_subd_to_grid_prefix_embedding_layers[i_subd], self.subd_to_content_context_embedding_layers[i_subd], context_content_indicator=1) # #### 

        #     #### subd_prefix_embedding for prefix vertices #### # subd_prefix_embedding
        #     cur_subd_prefix_embedding = self._embed_pts_with_contextual_information(cur_subd_prefix_embedding, context_cur_subd_prefix_embedding, edges=cur_subd_edges)

        #     cur_subd_prefix_embedding = cur_subd_prefix_embedding.contiguous().view(bsz, nn_subd_prefix_pts * 3, -1).contiguous() # ### 
        #     ### jerts! 
        


        # if self.training:
        #     print(f"cur_subd_upsample: {cur_subd_upsample.size()}, max_cur_subd_upsample: {cur_subd_upsample.max()}, min_cur_subd_upsample: {cur_subd_upsample.min()}")

        # if not self.training:
        #     print("here3!")
        # print(f" cur_subd_upsample: {cur_subd_upsample.size()}, max_cur_subd_upsample: {cur_subd_upsample.max()}, min_cur_subd_upsample: {cur_subd_upsample.min()}")
        ##### cur_subd_upsample_vert_embedding #####

        
        if not self.use_local_frame:
            cur_subd_upsample_vert_embedding = self._embed_pts(cur_subd_upsample, self.subd_to_grid_embedding_layers[str(i_subd)], self.subd_to_content_context_embedding_layers[i_subd], context_content_indicator=0)
            # if self.training:
            #     print(f"cur_subd_upsample_vert_embedding: {cur_subd_upsample_vert_embedding.size()}")
            # context_cur_subd_upsample_vert_embedding = self._embed_pts(cur_subd_upsample, self.context_subd_to_grid_embedding_layers[str(i_subd)], self.subd_to_content_context_embedding_layers[i_subd], context_content_indicator=1)
            # if self.training:
            #     print(f"context_cur_subd_upsample_vert_embedding: {context_cur_subd_upsample_vert_embedding.size()}")

            # cur_subd_upsample_vert_embedding = self._embed_pts_with_contextual_information(cur_subd_upsample_vert_embedding, context_cur_subd_upsample_vert_embedding, edges=nex_subd_edges) # 

        else: ### use local frame for vertices context embedding
            cur_subd_upsample_vert_embedding = self._embed_pts_with_contextual_information_local_frame(
                cur_subd_upsample, nex_subd_edges, i_subd=i_subd # 
            )

        # if self.training:
        #     print(f"cur_subd_upsample_vert_embedding: {cur_subd_upsample_vert_embedding.size()}")

        # cur_subd_upsample_vert_embedding: bsz x content_pts_nn x 3 x embedding_dim #
        cur_subd_upsample_vert_embedding = cur_subd_upsample_vert_embedding.contiguous().view(bsz, nn_verts * 3, -1).contiguous()

        # if self.training:
        #     print(f"cur_subd_upsample_vert_embedding: {cur_subd_upsample_vert_embedding.size()}")

        #### 
        # n_verts = cur_subd_upsample.size(1)
        # verts_order = torch.arange(n_verts).cuda()
        # verts_order_embedding = self.grid_embedding_layers[0](verts_order).unsqueeze(0) # bsz x n_verts x embedding_dim

        # xyz_indicator = torch.arange(3).cuda()
        # verts_xyz_indicator_embedding = self.grid_embedding_layers[-1](xyz_indicator).unsqueeze(0) # bsz x 3 x embedding_dim
        
        # cur_subd_upsample_vert_embedding = self.grid_embedding_layers[1](cur_subd_upsample) # bsz x n_verts x 3 x embedding_dim
        # # nex_subd_gt_vert_embedding = self.grid_embedding_layers[2](nex_subd_gt) # bsz x n_verts x 3 x embedding_dim
        # # print(f"vert embedding: {cur_subd_upsample_vert_embedding.size()}, xyz_indicator: {verts_xyz_indicator_embedding.size()}, order embedding: {verts_order_embedding.size()}")
        # cur_subd_upsample_vert_embedding = cur_subd_upsample_vert_embedding + verts_order_embedding.unsqueeze(2) + verts_xyz_indicator_embedding.unsqueeze(1) # bsz x n_verts x 3 x embedding_dim


        # bsz = cur_subd_upsample.size(0)
        # n_verts = cur_subd_upsample.size(1)
        # # bsz x (n_verts x 3) x embedding_dim
        # cur_subd_upsample_vert_embedding = cur_subd_upsample_vert_embedding.contiguous().view(bsz, n_verts * 3, -1).contiguous()

        # cur_subd_upsample_vert_embedding = torch.cat(
        #     [global_context_embedding, cur_subd_prefix_embedding, cur_subd_upsample_vert_embedding], dim=1
        # )
        cur_subd_upsample_vert_embedding = torch.cat(
            [global_context_embedding, cur_subd_upsample_vert_embedding], dim=1
        )

        # if self.training:
        #     print(f"cur_subd_upsample_vert_embedding: {cur_subd_upsample_vert_embedding.size()}")


        subd_to_verts_embedding[i_subd] = cur_subd_upsample_vert_embedding.cuda()
      return subd_to_verts_embedding


    def _embed_half_flaps(self, even_vertices_embeddings, half_flaps, half_flaps_embedding_layers):
        bsz, nn_flaps = half_flaps.size(0), half_flaps.size(1) # bsz x nn_flaps
        # half_flaps: bsz x (nn_edges x 2) x 4
        
        half_flaps_vertices_embeddings = batched_index_select(values=even_vertices_embeddings, indices=half_flaps, dim=1) ### bsz x (nn_edges x 2) x 4 x 3 x dim
        half_flaps_vertices_embeddings = half_flaps_vertices_embeddings[:, :, :, 0] + half_flaps_vertices_embeddings[:, :, :, 1] + half_flaps_vertices_embeddings[:, :, :, 2]
        #### half_flaps_vertices_embeddings: 
        half_flaps_vertices_embeddings = half_flaps_vertices_embeddings.contiguous().view(bsz, nn_flaps, -1).contiguous() ### bsz x (nn_edges x 2) x (4 x dim)
        half_flaps_vertices_embeddings = half_flaps_embedding_layers(half_flaps_vertices_embeddings) ### bsz x (nn_edges x 2) x dim
        half_flaps_vertices_embeddings = half_flaps_vertices_embeddings.unsqueeze(2).contiguous().repeat(1, 1, 3, 1).contiguous()
        
        # print(f"even_vertices_embeddings: {even_vertices_embeddings.size(), half_flaps_vertices_embeddings.size()}")
        even_vertices_aggregations = torch.zeros_like(even_vertices_embeddings) ### bsz x (nn_even_vertices) x dim
        even_vertices_aggregations[:, half_flaps[:, :, 0]] += half_flaps_vertices_embeddings
        even_vertices_aggregations[:, half_flaps[:, :, 1]] += half_flaps_vertices_embeddings
        even_vertices_aggregations[:, half_flaps[:, :, 2]] += half_flaps_vertices_embeddings
        even_vertices_aggregations[:, half_flaps[:, :, 3]] += half_flaps_vertices_embeddings
        even_vertices_aggregations /= 4.
        # even_vertices_aggregations = even_vertices_aggregations.unsqueeze(2)
        return half_flaps_vertices_embeddings, even_vertices_aggregations

    def _create_dist_grid_coord_v2(self, batch, subd_to_verts_embedding,  temperature=1., top_k=0, top_p=1.0, rt_logits=False):
        ### create dists accoridng to the embeddigns ###
    #   bsz, grid_length = grid_order_embedding.size(0), grid_order_coord_xyz_embeddings.size(1) - 1
      # grid_coord_outputs = self.decoder_grid_coord(grid_order_embedding, sequential_context_embeddings=grid_content_embedding)
      # dist grid v2

    #   prefix_length = cur_subd_prefix_pts_nn + self.prefix_key_len
      prefix_length = self.prefix_key_len
      # subd_to_coord_outputs = {}
      subd_to_dist_xyz = {}
      for i_subd in subd_to_verts_embedding:
        # cur_subd_prefix_pts_nn = batch[f'subd_{i_subd}'].size(1) * 3 # prefix_pts_nn x 3
        cur_subd_embedding = subd_to_verts_embedding[i_subd] # ### cur_subd_embedding ### cur_subd_
        
        cur_subd_decoder_grid_coord = self.subd_to_decoder_grid_coord[i_subd].cuda()

        #### absolute positional dependent ####
        if isinstance(cur_subd_decoder_grid_coord, TransformerDecoderGrid):
            grid_coord_outputs = cur_subd_decoder_grid_coord(cur_subd_embedding, sequential_context_embeddings=None, prefix_length=prefix_length) 
        elif isinstance(cur_subd_decoder_grid_coord, nn.Sequential):
            cur_subd_embedding = cur_subd_embedding[:, prefix_length: ]
            grid_coord_outputs = cur_subd_decoder_grid_coord(cur_subd_embedding)
        else:
            raise ValueError(f"Unrecognized deocder network: {type(cur_subd_decoder_grid_coord)}.")
        
        # grid_coord_outputs: bsz x (grid_length x 3) x embedding_dim
        grid_coord_outputs = grid_coord_outputs.contiguous().view(grid_coord_outputs.size(0), -1, 3, grid_coord_outputs.size(-1)).contiguous()
        cur_project_to_logits_layers = self.subd_to_grid_project_to_logits[i_subd]

        ##### use categorical distribution ####
        # logits_x, logits_y, logits_z = cur_project_to_logits_layers[0](grid_coord_outputs[:, :, 0]), cur_project_to_logits_layers[1](grid_coord_outputs[:, :, 1]), cur_project_to_logits_layers[2](grid_coord_outputs[:, :, 2])

        # #### use categorical distribution ####
        # logits_x /= temperature; logits_y /= temperature; logits_z /= temperature
        # logits_x = layer_utils.top_k_logits(logits_x, top_k)
        # logits_x = layer_utils.top_p_logits(logits_x, top_p)

        # logits_y = layer_utils.top_k_logits(logits_y, top_k)
        # logits_y = layer_utils.top_p_logits(logits_y, top_p)

        # logits_z = layer_utils.top_k_logits(logits_z, top_k) # bsz x grid_length x (1 + nn_grid_coord_discretization)
        # logits_z = layer_utils.top_p_logits(logits_z, top_p)

        # logits_xyz = torch.cat(
        #   [logits_x.unsqueeze(2), logits_y.unsqueeze(2), logits_z.unsqueeze(2)], dim=2
        # )
        # cat_dist_grid_xyz = torch.distributions.Categorical(logits=logits_xyz)
        # #### use categorical distribution ####

        if opt.model.pred_prob:
            logits_x, logits_y, logits_z = cur_project_to_logits_layers[0](grid_coord_outputs[:, :, 0]), cur_project_to_logits_layers[1](grid_coord_outputs[:, :, 1]), cur_project_to_logits_layers[2](grid_coord_outputs[:, :, 2])

            #### use categorical distribution ####
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
            cat_dist_grid_xyz = torch.distributions.Categorical(logits=logits_xyz)
            #### use categorical distribution ####

            subd_to_dist_xyz[i_subd] = cat_dist_grid_xyz
        else:
            # logits_x: bsz x nn_verts x dim
            logits_x, logits_y, logits_z = cur_project_to_logits_layers[0](grid_coord_outputs[:, :, 0]), cur_project_to_logits_layers[1](grid_coord_outputs[:, :, 1]), cur_project_to_logits_layers[2](grid_coord_outputs[:, :, 2])
            delta_x, delta_y, delta_z = logits_x[:, :, 0:1], logits_y[:, :, 0:1], logits_z[:, :, 0:1]
            # delta_x, delta_y, delta_z = logits_z[:, :, 0:1], logits_z[:, :, 1:2], logits_z[:, :, 2:3]
            cat_dist_grid_xyz = torch.cat(
                [delta_x, delta_y, delta_z], dim=-1 # bsz x nn_verts x 3
            )
            subd_to_dist_xyz[i_subd] = cat_dist_grid_xyz


        #### use normal distribution ####
        # logits_x, logits_y, logits_z = logits_x[:, :, :2], logits_y[:, :, :2], logits_z[:, :, :2]
        
        #### for each coordinate individually ####
        # mu_x, sigma_x = logits_x[:, :, 0], torch.exp(logits_x[:, :, 1])
        # mu_y, sigma_y = logits_y[:, :, 0], torch.exp(logits_y[:, :, 1])
        # mu_z, sigma_z = logits_z[:, :, 0], torch.exp(logits_z[:, :, 1])
        # mu_xyz = torch.cat(
        #     [mu_x.unsqueeze(-1), mu_y.unsqueeze(-1), mu_z.unsqueeze(-1)], dim=-1
        # )
        # # mu_xyz = torch.sigmoid(mu_xyz) - 0.5
        # sigma_xyz = torch.cat(
        #     [sigma_x.unsqueeze(-1), sigma_y.unsqueeze(-1), sigma_z.unsqueeze(-1)], dim=-1
        # )
        #### for each coordinate individually ####

        subd_to_dist_xyz[i_subd] = cat_dist_grid_xyz
        

        # subd_to_dist_xyz[i_subd] = cat_dist_grid_xyz
      return subd_to_dist_xyz

    
    def _create_dist_loop_subdiv(self, batch,  temperature=1., top_k=0, top_p=1.0, rt_logits=False, global_context_embedding=None):
        prefix_length = self.prefix_key_len
        # subd_to_coord_outputs = {}
        subd_to_dist_xyz = {}

        subdn = opt.dataset.subdn 

        sampled_subd_to_pts = {}
        
        for i_subd in range(subdn - 1): ###### subdn_to_embeddings
            if f'subd_{i_subd}' not in batch:
                continue
            # if not self.training:
            #     print("here...5")
            #### cur_subd_upsample ####
            #### upsample in the next level ####
            if f'subd_{i_subd}_upsample' in sampled_subd_to_pts: #### sampeled subd_to_pts ####
                cur_subd_upsample = sampled_subd_to_pts[f'subd_{i_subd}_upsample']
            else:  
                cur_subd_upsample = batch[f'subd_{i_subd}_upsample'] # bsz x n_verts x 3
            # cur_subd_edges = batch[f'subd_{i_subd}_edges'] # bsz x 2 x nn_edges_verts 
            nex_subd_edges = batch[f'subd_{i_subd}_upsample_edges']

            bsz, nn_verts = cur_subd_upsample.size(0), cur_subd_upsample.size(1)
            # nn_subd_prefix_pts = cur_subd_prefix_pts.size(1)
            
            if not self.use_local_frame:
                cur_subd_upsample_vert_embedding = self._embed_pts(cur_subd_upsample, self.subd_to_grid_embedding_layers[str(i_subd)], self.subd_to_content_context_embedding_layers[i_subd], context_content_indicator=0)
                # if self.training:
                #     print(f"cur_subd_upsample_vert_embedding: {cur_subd_upsample_vert_embedding.size()}")
                # context_cur_subd_upsample_vert_embedding = self._embed_pts(cur_subd_upsample, self.context_subd_to_grid_embedding_layers[str(i_subd)], self.subd_to_content_context_embedding_layers[i_subd], context_content_indicator=1)
                # if self.training:
                #     print(f"context_cur_subd_upsample_vert_embedding: {context_cur_subd_upsample_vert_embedding.size()}")

            else: ### use local frame for vertices context embedding
                cur_subd_upsample_vert_embedding = self._embed_pts_with_contextual_information_local_frame(
                    cur_subd_upsample, nex_subd_edges, i_subd=i_subd # 
                )

            # cur_subd_upsample_vert_embedding: bsz x content_pts_nn x 3 x embedding_dim #
            cur_subd_upsample_vert_embedding = cur_subd_upsample_vert_embedding.contiguous().view(bsz, nn_verts * 3, -1).contiguous()

            cur_subd_upsample_vert_embedding = torch.cat(
                [global_context_embedding, cur_subd_upsample_vert_embedding], dim=1
            )

            cur_subd_decoder_grid_coord = self.subd_to_decoder_grid_coord[i_subd].cuda()

            #### absolute positional dependent ####
            if isinstance(cur_subd_decoder_grid_coord, TransformerDecoderGrid):
                grid_coord_outputs = cur_subd_decoder_grid_coord(cur_subd_upsample_vert_embedding, sequential_context_embeddings=None, prefix_length=prefix_length) 
            elif isinstance(cur_subd_decoder_grid_coord, nn.Sequential):
                cur_subd_upsample_vert_embedding = cur_subd_upsample_vert_embedding[:, prefix_length: ]
                grid_coord_outputs = cur_subd_decoder_grid_coord(cur_subd_upsample_vert_embedding)
            else:
                raise ValueError(f"Unrecognized deocder network: {type(cur_subd_decoder_grid_coord)}.")
            
            # grid_coord_outputs: bsz x (grid_length x 3) x embedding_dim
            grid_coord_outputs = grid_coord_outputs.contiguous().view(grid_coord_outputs.size(0), -1, 3, grid_coord_outputs.size(-1)).contiguous()
            cur_project_to_logits_layers = self.subd_to_grid_project_to_logits[i_subd]

            if opt.model.pred_prob:
                logits_x, logits_y, logits_z = cur_project_to_logits_layers[0](grid_coord_outputs[:, :, 0]), cur_project_to_logits_layers[1](grid_coord_outputs[:, :, 1]), cur_project_to_logits_layers[2](grid_coord_outputs[:, :, 2])

                #### use categorical distribution ####
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
                cat_dist_grid_xyz = torch.distributions.Categorical(logits=logits_xyz)
                #### use categorical distribution ####

                subd_to_dist_xyz[i_subd] = cat_dist_grid_xyz
            else:
                # logits_x: bsz x nn_verts x dim
                logits_x, logits_y, logits_z = cur_project_to_logits_layers[0](grid_coord_outputs[:, :, 0]), cur_project_to_logits_layers[1](grid_coord_outputs[:, :, 1]), cur_project_to_logits_layers[2](grid_coord_outputs[:, :, 2])
                # delta_x, delta_y, delta_z = logits_x[:, :, 0:1], logits_y[:, :, 0:1], logits_z[:, :, 0:1]
                delta_x, delta_y, delta_z = logits_z[:, :, 0:1], logits_z[:, :, 1:2], logits_z[:, :, 2:3]
                cat_dist_grid_xyz = torch.cat(
                    [delta_x, delta_y, delta_z], dim=-1 # bsz x nn_verts x 3
                )
                subd_to_dist_xyz[i_subd] = cat_dist_grid_xyz

            if i_subd < subdn - 2:
                if opt.model.pred_prob:
                    cur_subd_pred_delta = cat_dist_grid_xyz.sample()
                    if opt.model.pred_delta:
                        cur_subd_pred_delta = data_utils.dequantize_verts_torch(cur_subd_pred_delta, n_bits=self.quantization_bits, min_range=opt.dataset.min_quant_range, max_range=opt.dataset.max_quant_range)    
                    else:
                        cur_subd_pred_delta = data_utils.dequantize_verts_torch(cur_subd_pred_delta, n_bits=self.quantization_bits)  
                else:
                    cur_subd_pred_delta = cat_dist_grid_xyz

                cur_subd_upsample = batch[f'subd_{i_subd}_upsample']
                #### dequantized delta positions ####
                if opt.model.pred_delta:   
                    cur_subd_pred_upsample = cur_subd_upsample + cur_subd_pred_delta ### subd_pred_delta
                else:
                    cur_subd_pred_upsample = cur_subd_pred_delta ### predict upsample vertices in the next subdivision level ###

                #### nex subdiv faces ####
                nex_subd_faces = batch[f'subd_{i_subd + 1}_faces']
                # selected_bfs_verts = batch[f'subd_{i_subd}_upsample_pts_indices']
                nex_subd_upsample_xyz = self.upsample_verts(cur_subd_pred_upsample, nex_subd_faces)
                # nex_subd_upsample_xyz = nex_subd_upsample_xyz[:, selected_bfs_verts] ### bsz x nn_verts x 3

                sampled_subd_to_pts[f'subd_{i_subd + 1}_upsample'] = nex_subd_upsample_xyz
                

            # subd_to_verts_embedding[i_subd] = cur_subd_upsample_vert_embedding.cuda()
        return subd_to_dist_xyz

    def _create_dist_loop_subdiv_v2(self, batch,  temperature=1., top_k=0, top_p=1.0, rt_logits=False, global_context_embedding=None):
        prefix_length = self.prefix_key_len
        # subd_to_coord_outputs = {}
        subd_to_dist_xyz = {}

        subdn = opt.dataset.subdn 

        sampled_subd_to_pts = {}
        
        for i_subd in range(subdn - 1): ###### subdn_to_embeddings
            if f'subd_{i_subd}' not in batch:
                continue
            # if not self.training:
            #     print("here...5")
            #### cur_subd_upsample ####
            # cur_subd_even_
            if f'subd_{i_subd}_upsample' in sampled_subd_to_pts:
                cur_subd_upsample = sampled_subd_to_pts[f'subd_{i_subd}_upsample'] #### upsampeld points
            else:  
                cur_subd_upsample = batch[f'subd_{i_subd}_upsample'] # bsz x n_verts x 3
            if f'subd_{i_subd}' in sampled_subd_to_pts:
                cur_subd = sampled_subd_to_pts[f'subd_{i_subd}']
            else:
                cur_subd = batch[f'subd_{i_subd}'] 
            # cur_subd_edges = batch[f'subd_{i_subd}_edges'] # bsz x 2 x nn_edges_verts 
            # nex_subd_edges = batch[f'subd_{i_subd}_upsample_edges']
            cur_subd_half_flaps = batch[f'subd_{i_subd}_half_flaps']
            cur_subd_half_flaps_mid_points_idx = batch[f'subd_{i_subd}_half_flaps_mid_points_idx']

            nn_cur_subd_verts = cur_subd.size(1)
            bsz, nn_verts = cur_subd_upsample.size(0), cur_subd_upsample.size(1)
            # nn_subd_prefix_pts = cur_subd_prefix_pts.size(1)
            
            # if not self.use_local_frame:
            #     cur_subd_upsample_vert_embedding = self._embed_pts(cur_subd_upsample, self.subd_to_grid_embedding_layers[str(i_subd)], self.subd_to_content_context_embedding_layers[i_subd], context_content_indicator=0)
            #     # if self.training:
            #     #     print(f"cur_subd_upsample_vert_embedding: {cur_subd_upsample_vert_embedding.size()}")
            #     # context_cur_subd_upsample_vert_embedding = self._embed_pts(cur_subd_upsample, self.context_subd_to_grid_embedding_layers[str(i_subd)], self.subd_to_content_context_embedding_layers[i_subd], context_content_indicator=1)
            #     # if self.training:
            #     #     print(f"context_cur_subd_upsample_vert_embedding: {context_cur_subd_upsample_vert_embedding.size()}")

            # else: ### use local frame for vertices context embedding
            #     cur_subd_upsample_vert_embedding = self._embed_pts_with_contextual_information_local_frame(
            #         cur_subd_upsample, nex_subd_edges, i_subd=i_subd # 
            #     )

            #### cur_subd_vert_embedding ####
            cur_subd_vert_embedding = self._embed_pts(cur_subd, self.subd_to_grid_prefix_embedding_layers[i_subd], self.subd_to_content_context_embedding_layers[i_subd], context_content_indicator=1)

            cur_subd_upsample_vert_embedding = self._embed_pts(cur_subd_upsample, self.subd_to_grid_embedding_layers[str(i_subd)], self.subd_to_content_context_embedding_layers[i_subd], context_content_indicator=0)

            # print()
            half_flaps_vertices_embeddings, even_vertices_aggregations = self._embed_half_flaps(cur_subd_vert_embedding, cur_subd_half_flaps, self.subd_to_half_flap_embedding_layers[i_subd])
            
            # # even_vertices_aggregations
            # print(f"cur_subd_upsample_vert_embedding: {cur_subd_upsample_vert_embedding.size()}")
            even_vertices_embeddigns = cur_subd_upsample_vert_embedding[:, : nn_cur_subd_verts]
            even_vertices_embeddigns = even_vertices_embeddigns + even_vertices_aggregations ### bsz x nn_even_verts x dim
            even_vertices_embeddigns = even_vertices_embeddigns.contiguous().view(bsz, even_vertices_embeddigns.size(1) * 3, -1).contiguous()
            # even_vertices_embeddigns = even_vertices_embeddigns[:, :, 0] + even_vertices_embeddigns[:, :, 1] + even_vertices_embeddigns[:, :, 2]
            # print(f"global_context_embedding: {global_context_embedding.size()}, even_vertices_embeddigns: {even_vertices_embeddigns.size()}")
            even_vertices_embeddigns = torch.cat(
                [global_context_embedding, even_vertices_embeddigns], dim=1
            )


            cur_subd_decoder_grid_coord = self.subd_to_decoder_grid_coord[i_subd].cuda()


            #### even_vertices_embeddigns
            even_vertices_embeddigns = cur_subd_decoder_grid_coord(even_vertices_embeddigns, sequential_context_embeddings=None, prefix_length=prefix_length)  
            #### even_vertices_embeddigns: bsz x nn_even_verts x 3 x dim
            even_vertices_embeddigns = even_vertices_embeddigns.contiguous().view(bsz, cur_subd_vert_embedding.size(1), 3, -1).contiguous()

            #### bsz x nn_half_flaps x dim
            half_flaps_vertices_embeddings, _ = self._embed_half_flaps(even_vertices_embeddigns, cur_subd_half_flaps, self.subd_to_half_flap_embedding_layers[i_subd])

            odd_vertices_embedding_aggregations = torch.zeros_like(cur_subd_upsample_vert_embedding[:, nn_cur_subd_verts: ]) ### bsz x nn_odd_vertices x dim

            # print(f"odd_vertices_embedding_aggregations: {odd_vertices_embedding_aggregations.size()}, cur_subd_half_flaps_mid_points_idx_max: {torch.max(cur_subd_half_flaps_mid_points_idx - nn_cur_subd_verts)}, cur_subd_half_flaps_mid_points_idx_min: {torch.min(cur_subd_half_flaps_mid_points_idx - nn_cur_subd_verts)}")
            odd_vertices_embedding_aggregations[:, cur_subd_half_flaps_mid_points_idx - nn_cur_subd_verts] += half_flaps_vertices_embeddings # .unsqueeze(2)

            odd_vertices_embedding_aggregations /= 2.
            odd_vertices_embeddings = cur_subd_upsample_vert_embedding[:, even_vertices_embeddigns.size(1): ] + odd_vertices_embedding_aggregations
            cur_subd_upsample_vert_embedding = torch.cat(
                [even_vertices_embeddigns, odd_vertices_embeddings], dim=1
            )
            
            # cur_subd_upsample_vert_embedding: bsz x content_pts_nn x 3 x embedding_dim #
            cur_subd_upsample_vert_embedding = cur_subd_upsample_vert_embedding.contiguous().view(bsz, nn_verts * 3, -1).contiguous()

            cur_subd_upsample_vert_embedding = torch.cat(
                [global_context_embedding, cur_subd_upsample_vert_embedding], dim=1
            )

        
            #### absolute positional dependent ####
            if isinstance(cur_subd_decoder_grid_coord, TransformerDecoderGrid):
                grid_coord_outputs = cur_subd_decoder_grid_coord(cur_subd_upsample_vert_embedding, sequential_context_embeddings=None, prefix_length=prefix_length) 
            elif isinstance(cur_subd_decoder_grid_coord, nn.Sequential):
                cur_subd_upsample_vert_embedding = cur_subd_upsample_vert_embedding[:, prefix_length: ]
                grid_coord_outputs = cur_subd_decoder_grid_coord(cur_subd_upsample_vert_embedding)
            else:
                raise ValueError(f"Unrecognized deocder network: {type(cur_subd_decoder_grid_coord)}.")
            
            # grid_coord_outputs: bsz x (grid_length x 3) x embedding_dim
            grid_coord_outputs = grid_coord_outputs.contiguous().view(grid_coord_outputs.size(0), -1, 3, grid_coord_outputs.size(-1)).contiguous()
            cur_project_to_logits_layers = self.subd_to_grid_project_to_logits[i_subd]

            logits_x, logits_y, logits_z = cur_project_to_logits_layers[0](grid_coord_outputs[:, :, 0]), cur_project_to_logits_layers[1](grid_coord_outputs[:, :, 1]), cur_project_to_logits_layers[2](grid_coord_outputs[:, :, 2])

            #### use categorical distribution ####
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
            cat_dist_grid_xyz = torch.distributions.Categorical(logits=logits_xyz)
            #### use categorical distribution ####

            subd_to_dist_xyz[i_subd] = cat_dist_grid_xyz

            if i_subd < subdn - 2 and self.training:
                cur_subd_pred_delta = cat_dist_grid_xyz.sample()
                cur_subd_upsample = cur_subd_upsample = batch[f'subd_{i_subd}_upsample']
                cur_subd_pred_delta = data_utils.dequantize_verts_torch(cur_subd_pred_delta, n_bits=self.quantization_bits, min_range=opt.dataset.min_quant_range, max_range=opt.dataset.max_quant_range)
                cur_subd_pred_upsample = cur_subd_upsample + cur_subd_pred_delta ### subd_pred_delta


                nex_subd_faces = batch[f'subd_{i_subd + 1}_faces']

                # selected_bfs_verts = batch[f'subd_{i_subd}_upsample_pts_indices']
                
                nex_subd_upsample_xyz = self.upsample_verts(cur_subd_pred_upsample, nex_subd_faces)
                # nex_subd_upsample_xyz = nex_subd_upsample_xyz[:, selected_bfs_verts] ### bsz x nn_verts x 3

                sampled_subd_to_pts[f'subd_{i_subd + 1}_upsample'] = nex_subd_upsample_xyz
                

            # subd_to_verts_embedding[i_subd] = cur_subd_upsample_vert_embedding.cuda()
        return subd_to_dist_xyz


    def _create_dist_grid_content(self, grid_embedding, grid_content_embedding, temperature=1., top_k=0, top_p=1.):
      # bsz, grid_length = grid_embedding.size(0), grid_embedding.size(1) - self.prefix_key_len

      # grid_content_outputs: bsz x (1 + grid_length) x embedding_dim
      # grid_content_outputs = self.decoder_grid_content(grid_embedding, sequential_context_embeddings=grid_content_embedding)
      # grid_content_outputs: bsz x grid_length x embedding_dim
      ##### grid content and dist ##### # grid 
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

    def extract_features(self, batch):
        ###### context features ######
        global_context, seq_context = self._prepare_context(batch)
        grid_xyzs = batch['grid_xyzs'] # n_grids x (3) --> grid_xyzs
        grid_content = batch['grid_content_vocab'] # use content_vocab for prediction and predict content_vocab
        if 'grid_pos' in batch:
            grid_pos = batch['grid_pos']
        else:
            grid_pos = None
        # grid_order_embedding, grid_embedding, grid_content_embedding: bsz x (1 + grid_length) x embedding_dim
        grid_order_embedding, grid_embedding, grid_content_embedding, grid_order_coord_xyz_embeddings = self._embed_input_grids(grid_xyzs=grid_xyzs,  grid_content=grid_content, grid_pos=grid_pos, global_context_embedding=global_context)

        #### decode features from coord_xyz_sequential embeddings ####
        #### grid_coord_outputs: bsz x seq_length x embedding_dim ####
        grid_coord_outputs = self.decoder_grid_coord(grid_order_coord_xyz_embeddings, sequential_context_embeddings=None, context_window=opt.model.context_window * 3) # only use coord for coord decoding
        
        # grid_coord_outputs: bsz x (grid_length x 3) x embedding_dim
        # grid_length = grid_coord_outputs.size(1) // 3
        # bsz = grid_coord_outputs.size(0)
        rep_features = grid_coord_outputs[:, -1, :]
        # grid_coord_outputs: bsz x grid_length x 3 x embedding_dim
        # grid_coord_outputs = grid_coord_outputs.contiguous().view(bsz, grid_length, 3, -1).contiguous()
        ######## rep_features: bsz x embedding_dim ########
        return rep_features

    def encode_and_sample(self, batch):
         ###### context features ######
        global_context, seq_context = self._prepare_context(batch)
        grid_xyzs = batch['grid_xyzs'] # n_grids x (3) --> grid_xyzs
        grid_content = batch['grid_content_vocab'] # use content_vocab for prediction and predict content_vocab
        if 'grid_pos' in batch:
            grid_pos = batch['grid_pos']
        else:
            grid_pos = None

        grid_length = grid_xyzs.size(1)
        delta_grid_size = 5
        # grid_xyzs: bsz x grid_length x 3
        for i_g in range(1, grid_length): # based on the first grid ## find the one that can maximize the distirubiton
            for c in range(3):
                #### dist_grid_coord
                grid_order_embedding, grid_embedding, grid_content_embedding, grid_order_coord_xyz_embeddings = self._embed_input_grids(grid_xyzs=grid_xyzs,  grid_content=grid_content, grid_pos=grid_pos, global_context_embedding=global_context)
                cat_dist_grid_xyz, logits_xyz = self._create_dist_grid_coord_v2(grid_order_coord_xyz_embeddings,  temperature=1., top_k=0, top_p=1.0, rt_logits=True)
                # logits_xyz: bsz x grid_length x 3 x logits_length
                # logits_xyz: bsz x
                cur_coord_logits = logits_xyz[:, i_g, c, :] # bsz x logits_length
                cur_real_coord = grid_xyzs[:, i_g, c] # bsz
                # cur_real_logit = data_utils.batched_index_select(values=cur_coord_logits, indices=cur_real_coord.unsqueeze(-1), dim=1) # bsz x 1
                cur_coord_arange = torch.arange(cur_coord_logits.size(-1), dtype=torch.long).cuda() # bsz x logits_length
                dist_real_coord_arange_coord = torch.abs(cur_real_coord.unsqueeze(-1) - cur_coord_arange.unsqueeze(0)) # bsz x logits_length
                cur_coord_logits[dist_real_coord_arange_coord >= delta_grid_size] = -1e7 # bsz x logits_length #### delta_grid_size
                cur_selected_coord = torch.argmax(cur_coord_logits, dim=-1) # bsz 
                grid_xyzs[:, i_g, c] = cur_selected_coord # bsz x grid_length x 3
            
            # grid_order_embedding, grid_embedding, grid_content_embedding: bsz x (1 + grid_length) x embedding_dim
            grid_order_embedding, grid_embedding, grid_content_embedding, grid_order_coord_xyz_embeddings = self._embed_input_grids(grid_xyzs=grid_xyzs,  grid_content=grid_content, grid_pos=grid_pos, global_context_embedding=global_context)
            pred_dist_grid_values = self._create_dist_grid_content(grid_embedding, grid_content_embedding)
            sampled_grid_values = pred_dist_grid_values.sample() # bsz x grid_length
            sampled_grid_values = sampled_grid_values[:, i_g]
            grid_content[:, i_g] = sampled_grid_values # sampled_grid_values # sampled_grid_values

        encoded_samples = batch
        encoded_samples['grid_xyzs'] = grid_xyzs
        encoded_samples['grid_values'] = grid_content
        
        deocded_samples = self.sample(grid_xyzs.size(0), context=encoded_samples, temperature=1., top_k=0, top_p=0.95, recenter_verts=True, cond_context_info=True, sampling_max_num_grids=-1)
        # encoded_samples = {
        #     'grid_xyzs': grid_xyzs - 1,
        #     'grid_values': grid_content
        # }
        return deocded_samples


    def forward_with_adaptation(self, batch, adapter_modules=None):
        
        global_context, seq_context = self._prepare_context(batch, adapter_module=adapter_modules)

        grid_xyzs = batch['grid_xyzs']
        grid_content = batch['grid_content_vocab']
        grid_xyzs_mask = batch['grid_xyzs_mask']

        if 'grid_pos' in batch:
            grid_pos = batch['grid_pos']
        else:
            grid_pos = None
        
        delta_grid_nn = 5
        # bsz, grid_length = grid_xyzs.size(0), grid_xyzs.size(1)
        # grid_order_embedding, grid_embedding, grid_content_embedding: bsz x (1 + grid_length) x embedding_dim
        grid_order_embedding, grid_embedding, grid_content_embedding, grid_order_coord_xyz_embeddings = self._embed_input_grids(grid_xyzs=grid_xyzs,  grid_content=grid_content, grid_pos=grid_pos, global_context_embedding=global_context)

        

        ### logits_xyz: bsz x n_grids x 3 x n_discrete_coordiantes ###
        cat_dist_grid_xyz, logits_xyz = self._create_dist_grid_coord_v2(grid_order_coord_xyz_embeddings,  temperature=1., top_k=0, top_p=1.0, rt_logits=True)

        n_discrete_grid_xyz = logits_xyz.size(-1)
        
        ### grid_xyzs: bsz x n_grids x 3 ###
        logits_arange = torch.arange(n_discrete_grid_xyz).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # bsz x n_grids x 3 x n_discrete_coordinates
        abs_dist_logits_arange_xyz = torch.abs(grid_xyzs.unsqueeze(-1) - logits_arange)
        # logits_xyzs: bsz x n_grids x 3 x n_val
        logits_xyz_tmp = logits_xyz.clone()
        logits_xyz_tmp[abs_dist_logits_arange_xyz >= delta_grid_nn] = -1e7

        
        ### maxx_grid_xyzs: bsz x n_grids x 3 ###
        maxx_grid_xyzs = torch.argmax(logits_xyz_tmp, dim=-1) #### maxx_grid_xyzs
        
        delta_xyz = maxx_grid_xyzs - grid_xyzs # bsz x n_grids x 3
        delta_xyz = delta_xyz.float() / float(delta_grid_nn - 1) # 
        
        loss_ori_grid_xyzs = -torch.sum( # ientifiers
            cat_dist_grid_xyz.log_prob(grid_xyzs) * grid_xyzs_mask, dim=-1
        ) 
        loss_ori_grid_xyzs = loss_ori_grid_xyzs.sum(-1) # .sum(-1) # bsz

        loss_maxx_grid_xyzs = -torch.sum( # ientifiers
            cat_dist_grid_xyz.log_prob(maxx_grid_xyzs) * grid_xyzs_mask, dim=-1
        ) 
        loss_maxx_grid_xyzs = loss_maxx_grid_xyzs.sum(-1)

        pred_dist_grid_values = self._create_dist_grid_content(grid_embedding, grid_content_embedding)

        grid_values_prediction_loss = -pred_dist_grid_values.log_prob(batch['grid_content_vocab']) * batch['grid_content_vocab_mask']
        grid_values_prediction_loss = grid_values_prediction_loss.sum(-1)

        # tot_prediction_loss = torch.mean(grid_values_prediction_loss + (loss_maxx_grid_xyzs + loss_ori_grid_xyzs) / 2.)
        return loss_ori_grid_xyzs, loss_maxx_grid_xyzs, grid_values_prediction_loss


    def upsample_verts(self, verts, faces):
        bsz = verts.size(0)
        tot_upsampled_verts = []
        for i_bsz in range(bsz):
            cur_bsz_faces_list = faces[i_bsz].detach().cpu().tolist()
            cur_bsz_verts = verts[i_bsz].detach().cpu().numpy()

            # cur_bsz_verts = data_utils.dequantize_verts(cur_bsz_verts, self.quantization_bits)

            cur_bsz_edge_list, cur_bsz_edge_to_exist = read_edges(cur_bsz_faces_list)
            upsampled_verts, _ = upsample_vertices(cur_bsz_verts, cur_bsz_edge_list)
            upsampled_verts = np.array(upsampled_verts, dtype=np.float)
            upsampled_verts = np.concatenate([cur_bsz_verts, upsampled_verts], axis=0)

            # upsampled_verts = data_utils.quantize_verts(upsampled_verts, n_bits=self.quantization_bits)

            # upsampled_verts = torch.from_numpy(upsampled_verts).long().cuda()
            upsampled_verts = torch.from_numpy(upsampled_verts).float().cuda()
            tot_upsampled_verts.append(upsampled_verts.unsqueeze(0))
        tot_upsampled_verts = torch.cat(tot_upsampled_verts, dim=0)
        return tot_upsampled_verts

    def get_sampled_edges(self, edges, old_idx_to_new_idx):
        new_edges = []
        bsz = edges.size(0)
        #### 
        # print("edges", edges.size())
        # print(f"sampling edges with edges: {edges.size()}")
        for i_bsz in range(bsz):
            cur_bsz_edges = edges[i_bsz]
            new_cur_bsz_edges = []
            for i_e in range(cur_bsz_edges.size(1)):
                cur_e = cur_bsz_edges[:, i_e].tolist()
                # print(cur_bsz_edges.size(), cur_e)
                cur_e_0, cur_e_1 = cur_e #### cur_e
                if cur_e_0 - 1 in old_idx_to_new_idx and cur_e_1 - 1 in old_idx_to_new_idx:
                    # cur_e_0 = old_idx_to_new_idx[cur_e_0 - 1]
                    # cur_e_1 = old_idx_to_new_idx[cur_e_1 - 1] # 
                    # new_cur_bsz_edges += [[cur_e_0 + 1, cur_e_1 + 1], [cur_e_1 + 1, cur_e_0 + 1]]
                    new_cur_e_0, new_cur_e_1 = old_idx_to_new_idx[cur_e_0 - 1] + 1, old_idx_to_new_idx[cur_e_1 - 1] + 1
                    new_cur_bsz_edges += [[new_cur_e_0, new_cur_e_1]]
            # print(f"new_cur_bsz_edges: {len(new_cur_bsz_edges)}")
            if len(new_cur_bsz_edges) == 0:
                new_cur_bsz_edges = torch.zeros((2, 0), dtype=torch.long).unsqueeze(0).cuda()
            else:
                new_cur_bsz_edges = torch.tensor(new_cur_bsz_edges, dtype=torch.long).transpose(0, 1).unsqueeze(0).cuda()
            new_edges.append(new_cur_bsz_edges)
        new_edges = torch.cat(new_edges, dim=0)
        return new_edges
    
    
    
    #### recenter_vertices
    def recenter_vertices(self, verts_gt, verts_upsample):
      verts_gt, verts_upsample = verts_gt.detach().cpu().numpy(), verts_upsample.detach().cpu().numpy()

    #   dequan_verts_gt, dequan_verts_upsample = data_utils.dequantize_verts(verts_gt, n_bits=self.quantization_bits), data_utils.dequantize_verts(verts_upsample, n_bits=self.quantization_bits)

      dequan_verts_gt = verts_gt
      dequan_verts_upsample = verts_upsample

      center_dequan_verts_gt = data_utils.get_batched_vertices_center(dequan_verts_gt) # bsz x 1 x 3
      dequan_verts_gt = dequan_verts_gt - center_dequan_verts_gt
      dequan_verts_upsample = dequan_verts_upsample - center_dequan_verts_gt

    #   verts_gt = data_utils.quantize_verts(dequan_verts_gt, n_bits=self.quantization_bits)
    #   verts_upsample = data_utils.quantize_verts(dequan_verts_upsample, n_bits=self.quantization_bits)
      
      verts_gt = dequan_verts_gt
      verts_upsample = dequan_verts_upsample

    #   verts_gt = torch.from_numpy(verts_gt).long().cuda()
    #   verts_upsample = torch.from_numpy(verts_upsample).long().cuda()

      verts_gt = torch.from_numpy(verts_gt).float().cuda()
      verts_upsample = torch.from_numpy(verts_upsample).float().cuda()
      return verts_gt, verts_upsample, center_dequan_verts_gt


    #### sample #### sample forward
    def sample_forward(self, batch, adapter_modules=None):
        global_context, seq_context = self._prepare_context(batch, adapter_module=adapter_modules)

        # subd_to_verts_embedding = self._embed_input(batch, global_context_embedding=global_context)
        
        subdn = opt.dataset.subdn  if self.training else opt.dataset.subdn_test
        # sample_batch = {f'subd_{0}'}
        # base_subd = 0
        base_subd = self.st_subd_idx
        sample_batch = {}
        sample_batch[f'subd_{base_subd}'] = batch[f'subd_{base_subd}']
        sample_batch[f'subd_{base_subd}_upsample'] = batch[f'subd_{base_subd}_upsample']
        sample_batch[f'subd_{base_subd}_edges'] = batch[f'subd_{base_subd}_edges']
        sample_batch[f'subd_{base_subd}_upsample_edges'] = batch[f'subd_{base_subd}_upsample_edges']
        # sample_batch[f'subd_{base_subd}'] = batch[f'subd_{base_subd}']

        subd_idxes = range(self.st_subd_idx, subdn - 1)
        # for i_subd in range(subdn - 1): #### subdn - 1 ####
        for i_subd in subd_idxes:
            sample_batch[f'subd_{i_subd + 1}_edges'] = batch[f'subd_{i_subd + 1}_edges']
            sample_batch[f'subd_{i_subd}_upsample_edges'] = batch[f'subd_{i_subd}_upsample_edges']
            sample_batch[f'subd_{i_subd}_half_flaps'] = batch[f'subd_{i_subd}_half_flaps']
            sample_batch[f'subd_{i_subd}_half_flaps_mid_points_idx'] = batch[f'subd_{i_subd}_half_flaps_mid_points_idx']
            # sample_batch[f'subd_{i_subd + 1}_edges'] = batch[f'subd_{i_subd + 1}_edges']

            #### both previous verts and upsampled verts should be sub-indexed ####
            if sample_batch[f'subd_{i_subd}_upsample'].size(1) < self.max_num_grids: #### 
                # print(f"her sampling... with {i_subd}")
                select_batch = sample_batch
                cur_subd_upsample_verts = select_batch[f'subd_{i_subd}_upsample']
                # select_batch = batch

                if not opt.model.use_half_flaps:
                    subd_to_verts_embedding = self._embed_input(select_batch, global_context_embedding=global_context)
                    subd_to_dist_xyz = self._create_dist_grid_coord_v2(select_batch, subd_to_verts_embedding,  temperature=1., top_k=0, top_p=1.0, rt_logits=False)
                else:
                    subd_to_dist_xyz = self._create_dist_loop_subdiv_v2(select_batch,  temperature=1., top_k=0, top_p=1.0, rt_logits=False, global_context_embedding=global_context)

                ##### sample forward #####
                if isinstance(subd_to_dist_xyz[i_subd], torch.distributions.Categorical):
                    cur_subd_xyz_dist = subd_to_dist_xyz[i_subd]
                    cur_subd_delta_xyz = cur_subd_xyz_dist.sample() # bsz x n_verts x 3
                    # cur_subd_upsample_xyz
                    if opt.model.pred_delta:
                        cur_subd_delta_xyz = data_utils.dequantize_verts_torch(cur_subd_delta_xyz, n_bits=self.quantization_bits, min_range=opt.dataset.min_quant_range, max_range=opt.dataset.max_quant_range)
                        cur_subd_upsample_xyz = cur_subd_delta_xyz +  cur_subd_upsample_verts
                    else:
                        cur_subd_delta_xyz = data_utils.dequantize_verts_torch(cur_subd_delta_xyz, n_bits=self.quantization_bits)
                        cur_subd_upsample_xyz = cur_subd_delta_xyz #### not pred delta ####
                elif isinstance(subd_to_dist_xyz[i_subd], torch.Tensor):
                    cur_subd_xyz_dist = subd_to_dist_xyz[i_subd]
                    cur_subd_delta_xyz = cur_subd_xyz_dist
                    if opt.model.pred_delta:
                        cur_subd_upsample_xyz = cur_subd_delta_xyz +  cur_subd_upsample_verts
                    else:
                        cur_subd_upsample_xyz = cur_subd_delta_xyz
                # elif isinstance(subd_to_dist_xyz[i_subd], torch.Tensor):
                elif isinstance(subd_to_dist_xyz[i_subd], dict):
                    cur_subd_delta_xyz = subd_to_dist_xyz[i_subd]['delta_xyz']
                    cur_subd_delta_xyz = cur_subd_delta_xyz.detach().cpu().numpy()

                    cur_subd_upsample_xyz = select_batch[f'subd_{i_subd}_upsample'].detach().cpu().numpy()
                    # cur_subd_upsample_xyz = data_utils.dequantize_verts(cur_subd_upsample_xyz, n_bits=self.quantization_bits)

                    cur_subd_upsample_xyz = cur_subd_upsample_xyz + cur_subd_delta_xyz


                    # cur_subd_upsample_xyz = data_utils.quantize_verts(cur_subd_upsample_xyz, n_bits=self.quantization_bits)
                    
                    # cur_subd_upsample_xyz = torch.from_numpy(cur_subd_upsample_xyz).long().cuda() ##### upsampled point

                    cur_subd_upsample_xyz = torch.from_numpy(cur_subd_upsample_xyz).cuda() ##### upsampled point

                    # cur_subd_upsample_xyz = torch.clamp(cur_subd_upsample_xyz, min=0, max=2 ** self.quantization_bits - 1)  
            else:
                print(f"here with subd: {i_subd}")
                fix_bsz = 0
                sample_batch[f'subd_{i_subd + 1}_edges'] = batch[f'subd_{i_subd + 1}_edges']
                # print(f"edges in the batch: {batch[f'subd_{i_subd}_edges'] .size()}")

                bsz = batch[f'subd_{base_subd}'].size(0)
                tot_upsample_xyz = []
                for i_bsz in range(bsz):
                    n_pts = sample_batch[f'subd_{i_subd}_upsample'][i_bsz].size(0) # #### npts of the current batch
                    upsampled_vert_idx_to_value = {}
                    select_probs = torch.ones((n_pts,), dtype=torch.float32).cuda() # float tensor and to cuda
                    # select_probs = select_probs / torch.sum(select_probs).item() # for select_probs
                    
                    cur_subd_upsample_faces = batch[f'subd_{i_subd + 1}_faces'][i_bsz].detach().cpu().tolist() # #### face list
                    cur_subd = batch[f'subd_{i_subd}'][i_bsz]
                    cur_subd_upsample_verts = sample_batch[f'subd_{i_subd}_upsample'][i_bsz]
                    
                    
                    # print(f"cur_subd_upsample_verts: {cur_subd_upsample_verts.size()}, max_cur_subd_upsample_verts: {cur_subd_upsample_verts.max()}. min_cur_subd_upsample_verts: {cur_subd_upsample_verts.min()}")
                    # sample_bfs_component(selected_vert, faces, max_num_grids)
                    # select_faces_via_verts(selected_verts, faces)
                    cur_subd_pred_nex_verts = cur_subd_upsample_verts.clone()
                    while len(upsampled_vert_idx_to_value) < n_pts:
                        # print(f"len(upsampled_vert_idx_to_value): {len(upsampled_vert_idx_to_value)}, n_pts: {n_pts}")
                        selected_vert = torch.distributions.Categorical(logits=select_probs).sample().item()
                        
                        ### bfs sampled verts ###
                        sampled_verts = data_utils.sample_bfs_component(selected_vert, cur_subd_upsample_faces, self.max_num_grids)
                        ### sorted sampled verts ###
                        sampled_verts = sorted(sampled_verts, reverse=False)
                        # print(f"len of sampled_verts: {len(sampled_verts)}, sampled_verts: {sampled_verts[:10]}")
                        ### sorted sampled verts of the previous level ###
                        prev_sampled_verts = [v for v in sampled_verts if v < cur_subd.size(0)] ## prev verts
                        old_idx_to_new_idx = {v: ii for ii, v in enumerate(sampled_verts)}
                        prev_old_idx_to_new_idx = {v: ii for ii, v in enumerate(prev_sampled_verts)}
                        
                        ### edges and prev old_idx_to_new_idx ###
                        cur_patch_prev_edges = self.get_sampled_edges(sample_batch[f'subd_{i_subd}_edges'], prev_old_idx_to_new_idx)
                        cur_patch_edges = self.get_sampled_edges(sample_batch[f'subd_{i_subd + 1}_edges'], old_idx_to_new_idx)
                        cur_patch_upsample_edges = self.get_sampled_edges(sample_batch[f'subd_{i_subd}_upsample_edges'], old_idx_to_new_idx)



                        # sampled_verts = torch.tensor(sampled_verts, dtype=torch.long).cuda()
                        # prev_sampled_verts = torch.tensor(prev_sampled_verts, dtype=torch.long).cuda()

                        sampled_verts = torch.tensor(sampled_verts, dtype=torch.long).cuda()
                        prev_sampled_verts = torch.tensor(prev_sampled_verts, dtype=torch.long).cuda()
                        cur_patch_verts_upsample = cur_subd_upsample_verts[sampled_verts]
                        cur_patch_verts = cur_subd[prev_sampled_verts]
                        # subd_to_verts_embedding = self._embed_input(batch, global_context_embedding=global_context)
                # subd_to_dist_xyz = self._create_dist_grid_coord_v2(batch, subd_to_verts_embedding,  temperature=1., top_k=0, top_p=1.0, rt_logits=False)
                        cur_patch_batch = {}
                        for prv_subd in range(self.st_subd_idx, i_subd - 1):
                            cur_patch_batch[f'subd_{prv_subd}'] = sample_batch[f'subd_{prv_subd}'][i_bsz].unsqueeze(0)
                            cur_patch_batch[f'subd_{prv_subd}_upsample'] = sample_batch[f'subd_{prv_subd}_upsample'][i_bsz].unsqueeze(0)
                        
                        cur_patch_verts = cur_patch_verts.unsqueeze(0)
                        cur_patch_verts_upsample = cur_patch_verts_upsample.unsqueeze(0)

                        cur_patch_verts_upsample, cur_patch_verts, center_dequan_verts_gt = self.recenter_vertices(cur_patch_verts_upsample, cur_patch_verts)

                        cur_patch_batch[f'subd_{i_subd}'] = cur_patch_verts # cur_patch_verts 
                        cur_patch_batch[f'subd_{i_subd}_upsample'] = cur_patch_verts_upsample # cur_patch_verts_upsample

                        cur_patch_batch[f'subd_{i_subd}_edges'] = cur_patch_prev_edges[i_bsz].unsqueeze(0)
                        cur_patch_batch[f'subd_{i_subd + 1}_edges'] = cur_patch_edges[i_bsz].unsqueeze(0)
                        cur_patch_batch[f'subd_{i_subd}_upsample_edges'] = cur_patch_upsample_edges[i_bsz].unsqueeze(0)

                        # # print(cur_patch_verts.size(), cur_patch_verts_upsample.size())

                        if not opt.model.use_half_flaps:
                            subd_to_verts_embedding = self._embed_input(cur_patch_batch, global_context_embedding=global_context, i_subd=i_subd)
                            subd_to_dist_xyz = self._create_dist_grid_coord_v2(cur_patch_batch, subd_to_verts_embedding,  temperature=1., top_k=0, top_p=1.0, rt_logits=False)
                        else:
                            subd_to_dist_xyz = self._create_dist_loop_subdiv_v2(cur_patch_batch,  temperature=1., top_k=0, top_p=1.0, rt_logits=False, global_context_embedding=global_context)

                        if isinstance(subd_to_dist_xyz[i_subd], torch.distributions.Categorical):
                            cur_subd_xyz_dist = subd_to_dist_xyz[i_subd]
                            #### upsample xyzs ####
                            cur_subd_upsample_xyz = cur_patch_batch[f'subd_{i_subd}_upsample']
                            #### sample and dequantize for delta xyzs ####
                            cur_subd_delta_xyz = cur_subd_xyz_dist.sample() # bsz x n_verts x 3

                            if opt.model.pred_delta:
                                cur_subd_delta_xyz = data_utils.dequantize_verts_torch(cur_subd_delta_xyz, n_bits=self.quantization_bits, min_range=opt.dataset.min_quant_range, max_range=opt.dataset.max_quant_range) # bsz x n_verts x 3
                                #### delta xyzs ####
                                cur_subd_upsample_xyz = cur_subd_delta_xyz + torch.from_numpy(center_dequan_verts_gt).float().cuda() + cur_subd_upsample_xyz
                            else:
                                cur_subd_upsample_xyz = cur_subd_delta_xyz + torch.from_numpy(center_dequan_verts_gt).float().cuda()
                        elif isinstance(subd_to_dist_xyz[i_subd], torch.Tensor):
                            cur_subd_xyz_dist = subd_to_dist_xyz[i_subd]
                            cur_subd_delta_xyz = cur_subd_xyz_dist
                            if opt.model.pred_delta:
                                cur_subd_upsample_xyz = cur_subd_delta_xyz +  cur_subd_upsample_verts + torch.from_numpy(center_dequan_verts_gt).float().cuda()
                            else:
                                cur_subd_upsample_xyz = cur_subd_delta_xyz + torch.from_numpy(center_dequan_verts_gt).float().cuda()
                        # elif isinstance(subd_to_dist_xyz[i_subd], torch.Tensor):
                        elif isinstance(subd_to_dist_xyz[i_subd], dict):
                            cur_subd_delta_xyz = subd_to_dist_xyz[i_subd]['delta_xyz']
                            cur_subd_delta_xyz = cur_subd_delta_xyz.detach().cpu().numpy()

                            cur_subd_upsample_xyz = cur_patch_batch[f'subd_{i_subd}_upsample'].detach().cpu().numpy()
                            # cur_subd_upsample_xyz = data_utils.dequantize_verts(cur_subd_upsample_xyz, n_bits=self.quantization_bits)

                            cur_subd_upsample_xyz = cur_subd_upsample_xyz + cur_subd_delta_xyz + center_dequan_verts_gt # bsz x 1 x 3
                            # cur_subd_upsample_xyz =  cur_subd_delta_xyz + center_dequan_verts_gt # bsz x 1 x 3
                            print(f"cur_subd_delta_xyz: {np.mean(np.abs(cur_subd_delta_xyz), axis=1)}")

                            # cur_subd_upsample_xyz = data_utils.quantize_verts(cur_subd_upsample_xyz, n_bits=self.quantization_bits)

                            # cur_subd_upsample_xyz = torch.from_numpy(cur_subd_upsample_xyz).long().cuda() ##### upsampled point
                            cur_subd_upsample_xyz = torch.from_numpy(cur_subd_upsample_xyz).cuda() ##### upsampled point

                            # cur_subd_upsample_xyz = torch.clamp(cur_subd_upsample_xyz, min=0, max=2 ** self.quantization_bits - 1)  
                        #### cur_subd_upsample_xyz #### ### 
                        cur_subd_pred_nex_verts[sampled_verts] = cur_subd_upsample_xyz[0]
                        select_probs[sampled_verts] = -1e5
                        # cur_patch_verts = cur_subd[prev_sampled_verts]
                        for i_pt in range(sampled_verts.size(0)):
                            upsampled_vert_idx_to_value[sampled_verts[i_pt].item()] = 1.
                    tot_upsample_xyz.append(cur_subd_pred_nex_verts.unsqueeze(0))
                cur_subd_upsample_xyz = torch.cat(tot_upsample_xyz, dim=0)
                    
            
            nex_subd_faces = batch[f'subd_{i_subd + 1}_faces']
            # print(f"subd: {i_subd}, cur_subd_upsample_xyz: {cur_subd_upsample_xyz.size()}, max_nex_subd_faces: {nex_subd_faces.max()}, min_nex_subd_faces: {nex_subd_faces.min()}")
            sample_batch[f'subd_{i_subd + 1}'] = cur_subd_upsample_xyz #### subd_upsample_xyz
            # print(f"cur_subd_upsample_xyz: {cur_subd_upsample_xyz.size()}")
            # if i_subd < subdn - 2:
            nex_subd_upsample_xyz = self.upsample_verts(cur_subd_upsample_xyz, nex_subd_faces)
            
            sample_batch[f'subd_{i_subd + 1}_upsample'] = nex_subd_upsample_xyz

            nex_subd_edges, _ = read_edges(nex_subd_faces[0].detach().cpu().tolist())
            nex_subd_edges = torch.tensor(nex_subd_edges, dtype=torch.long).cuda().unsqueeze(0)
            sample_batch[f'subd_{i_subd + 1}_edges'] = nex_subd_edges.contiguous().transpose(1, 2).contiguous()
            # print(f"subd: {i_subd}, max_cur_subd_upsample_xyz: {torch.max(cur_subd_upsample_xyz)}, min_cur_subd_upsample_xyz: {torch.min(cur_subd_upsample_xyz)}, max_nex_subd_upsample_xyz: {torch.max(nex_subd_upsample_xyz)}, min_nex_subd_upsample_xyz: {torch.min(nex_subd_upsample_xyz)}")
        return sample_batch
            
        
    def forward(self, batch, adapter_modules=None):
        global_context, seq_context = self._prepare_context(batch, adapter_module=adapter_modules)
        # [global_context_embedding, global_context_embedding_content, global_context_embedding_grid] = 
        # left_cond, rgt_cond = self._prepare_prediction(global_context) # _create_dist() #
        # print(f"vertices flat: {batch['vertices_flat'].size()}")

        #### vertices embeddings ####
        # pred_dist, outputs = self._create_dist(batch['vertices_flat'][:, :-1], embedding_layers=self.inputs_embedding_layers, project_to_logits_layer=self.project_to_logits, global_context_embedding=global_context, sequential_context_embeddings=seq_context,
        # )
        ######## subd_to_verts_embedding ########

        # for i_subd in range(opt.dataset.subdn - 1):
        #     cur_subd = batch[f'subd_{i_subd}']
        #     cur_subd_upsample = batch[f'subd_{i_subd}_upsample']
        #     print(f"i_subd: {i_subd}, cur_subd: {cur_subd.size()}, cur_subd_upsample: {cur_subd_upsample.size()}, max_cur_subd: {cur_subd.max()}, min_cur_subd: {cur_subd.min()}, max_cur_subd_upsample: {cur_subd_upsample.max()}, min_cur_subd_upsample: {cur_subd_upsample.min()}")


        ##### embed_inputs #####
        # subd_to_verts_embedding = self._embed_input(batch, global_context_embedding=global_context)
        ##### embed_inputs #####
        
        # # print(f"after vertices embedding with subd_to_verts_embedding: {len(subd_to_verts_embedding)}")

        ##### create distributions #####
        # subd_to_dist_xyz = self._create_dist_grid_coord_v2(batch, subd_to_verts_embedding,  temperature=1., top_k=0, top_p=1.0, rt_logits=False)
        ##### create distributions #####

        if not opt.model.use_half_flaps:
            subd_to_dist_xyz = self._create_dist_loop_subdiv(batch,  temperature=1., top_k=0, top_p=1.0, rt_logits=False, global_context_embedding=global_context)
        else:
            subd_to_dist_xyz = self._create_dist_loop_subdiv_v2( batch,  temperature=1., top_k=0, top_p=1.0, rt_logits=False, global_context_embedding=global_context)
        
        return subd_to_dist_xyz

        # subdn = opt.dataset.subdn
        
        # for i_subd in range(subdn - 1):
        #   cur_subd_upsample = batch[f'subd_{i_subd}_upsample']
        #   nex_subd_gt = batch[f'subd_{i_subd}_gt']
          

        # grid_xyzs = batch['grid_xyzs'] # n_grids x (3) --> grid_xyzs
        # grid_content = batch['grid_content_vocab'] # use content_vocab for prediction and predict content_vocab
        # if 'grid_pos' in batch:
        #     grid_pos = batch['grid_pos']
        # else:
        #     grid_pos = None
        
        # # bsz, grid_length = grid_xyzs.size(0), grid_xyzs.size(1)
        # # grid_order_embedding, grid_embedding, grid_content_embedding: bsz x (1 + grid_length) x embedding_dim
        # grid_order_embedding, grid_embedding, grid_content_embedding, grid_order_coord_xyz_embeddings = self._embed_input_grids(grid_xyzs=grid_xyzs,  grid_content=grid_content, grid_pos=grid_pos, global_context_embedding=global_context)

        # # print(f"grid_order_coord_xyz_embeddings: {grid_order_coord_xyz_embeddings.size()}")
        # # pred_dist_grid_xyz = self._create_dist_grid_coord(grid_order_embedding, grid_content_embedding)
        # pred_dist_grid_xyz = self._create_dist_grid_coord_v2(grid_order_coord_xyz_embeddings,  temperature=1., top_k=0, top_p=1.0)
        # pred_dist_grid_values = self._create_dist_grid_content(grid_embedding, grid_content_embedding)
        # pred_dist_grid_values = self._create_dist_grid_content_v2(grid_embedding, grid_content_embedding, grid_content, is_training=True)

        # pred_dist_grid_xyz, pred_dist_grid_values = self._create_dist_grid(batch, global_context_embedding=global_context, temperature=1., top_k=0, top_p=1.0)

        # return pred_dist_grid_xyz, pred_dist_grid_values

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

    ### from a certian, and just train on and the offsset should be a normal distribution...
    def _loop_sample_grid(self, loop_idx, samples, global_context_embedding=None, sequential_context_embeddings=None, temperature=1., top_k=0, top_p=1.0): # loop sample grids
      grid_xyzs = samples['grid_xyzs'] # bsz x (1 + cur_seq_length) x 3 #### gird_xyzs
      grid_content = samples['grid_content_vocab'] # bsz x (1 + cur_seq_length) x (gs x gs x gs)
      grid_pos = samples['grid_pos']

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
      
      ##### grid_xyzs_in #####
      ##### sample x ##### #### embed input grids...
      grid_order_embedding, grid_embedding, grid_content_embedding, grid_order_coord_xyz_embeddings = self._embed_input_grids(grid_xyzs=grid_xyzs_in,  grid_content=grid_content_in,  grid_pos=grid_pos_in, global_context_embedding=global_context_embedding)
    #   grid_order_embedding, grid_embedding, grid_content_embedding, grid_order_coord_xyz_embeddings = self._embed_input_grids(grid_xyzs=grid_xyzs_in,  grid_content=grid_content_in, global_context_embedding=global_context_embedding)
    
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

        accum_equal_zero = torch.sum(equal_to_zero, dim=-1)
        not_stop = (accum_equal_zero == 0).long() # 
        not_stop = torch.sum(not_stop).item() > 0 # not stop
        return (not not_stop)

    def _stop_cond_grid(self, samples):
        if self.multi_part_not_ar_object: # #### 
            equal_to_zero = (samples == 0).long() # equal to zero
            accum_equal_zero = torch.sum(equal_to_zero, dim=-1) ##### number of xyzs equal to 0
            # not_stop = (accum_equal_zero == 0).long() # 
            # not_stop = (accum_equal_zero <= 2).long() # not stop
            stop_indicator = (accum_equal_zero == 3).long() # all three euqalt to zero --> stop
            stop_indicator = torch.any(stop_indicator, dim=-1).long() # bsz; for each batch size
            not_stop = torch.sum(1 - stop_indicator).item() > 0
        else:
            if self.ar_object and self.num_objects > 1:
                equal_to_zero = (samples == 0).long() # equal to zero
                accum_equal_zero = torch.sum(equal_to_zero, dim=-1) # bsz x n_sampled_grids
                part_sep_indicator = (accum_equal_zero == 3).long() # bsz x n_sampled_grids
                stop_indicator = (torch.sum(part_sep_indicator, dim=-1) >= self.num_objects).long() # bsz
                not_stop = torch.sum(1 - stop_indicator).item() > 0 # at least one sample cannnot stop
            else:
                # print(f"in stop condition, samples: {samples}")
                equal_to_zero = (samples == 0).long() # equal to zero

                accum_equal_zero = torch.sum(equal_to_zero.sum(-1), dim=-1)
                not_stop = (accum_equal_zero == 0).long() #
                not_stop = torch.sum(not_stop).item() > 0 # not stop
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
        print(f"Cond context info! grid_xyzs: {grid_xyzs.size()}, grid_content: {grid_content.size()}, grid_pos: {grid_pos.size()}")
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
      

    def _sample_vertices(self, num_samples, context, context_feature): # sample
        loop_idx = 0
        # init samples
        # samples = torch.zeros([num_samples, 0], dtype=torch.long).cuda()
        # probs = torch.ones([num_samples, 0], dtype=torch.float32).cuda()
        sample_batch = {}
        for i_subd in range(opt.dataset.subdn - 1):
            cur_subd_upsample = context[f'subd_{i_subd}'] # bsz x n_pts x 3
            # cur_subd_upsample = torch.zeros([num_samples, 0], dtype=torch.long)
            cur_subd_gt = torch.zeros([num_samples, 1, 3], dtype=torch.long).cuda()
            sample_batch[f'subd_{i_subd + 1}_gt'] = cur_subd_gt
            sample_batch[f'subd_{i_subd}'] = cur_subd_upsample
            
        nn_sampled = 0
        while True: # _loop_sample for samples
            subd_to_verts_embedding = self._embed_input(sample_batch, global_context_embedding=context_feature)

            subd_to_dist_xyz = self._create_dist_grid_coord_v2(context, subd_to_verts_embedding,  temperature=1., top_k=0, top_p=1.0, rt_logits=False)

            for i_subd in subd_to_dist_xyz:
                cur_sample = subd_to_dist_xyz[i_subd].sample() # bsz x n_verts x 3
                # cur_sample = cur_sample[:, -1:]
                sample_batch[f'subd_{i_subd + 1}_gt'][:, -1, 0] = cur_sample[:, -1, 0]
                # sample_batch[f'subd_{i_subd + 1}_gt'] = torch.cat([sample_batch[f'subd_{i_subd + 1}_gt'], cur_sample], dim=1) # num_samples x (n_verts + 1) x 3

            subd_to_verts_embedding = self._embed_input(sample_batch, global_context_embedding=context_feature)
            subd_to_dist_xyz = self._create_dist_grid_coord_v2(context, subd_to_verts_embedding,  temperature=1., top_k=0, top_p=1.0, rt_logits=False)

            for i_subd in subd_to_dist_xyz:
                cur_sample = subd_to_dist_xyz[i_subd].sample() # bsz x n_verts x 3
                # cur_sample = cur_sample[:, -1:]
                sample_batch[f'subd_{i_subd + 1}_gt'][:, -1, 1] = cur_sample[:, -1, 1]
                # sample_batch[f'subd_{i_subd + 1}_gt'] = torch.cat([sample_batch[f'subd_{i_subd + 1}_gt'], cur_sample], dim=1) # num_samples x (n_verts + 1) x 3

            
            subd_to_verts_embedding = self._embed_input(sample_batch, global_context_embedding=context_feature)
            subd_to_dist_xyz = self._create_dist_grid_coord_v2(context, subd_to_verts_embedding,  temperature=1., top_k=0, top_p=1.0, rt_logits=False)

            for i_subd in subd_to_dist_xyz:
                cur_sample = subd_to_dist_xyz[i_subd].sample() # bsz x n_verts x 3
                # cur_sample = cur_sample[:, -1:]
                sample_batch[f'subd_{i_subd + 1}_gt'][:, -1, 2] = cur_sample[:, -1, 2]
                # sample_batch[f'subd_{i_subd + 1}_gt'] = torch.cat([sample_batch[f'subd_{i_subd + 1}_gt'], cur_sample], dim=1) # num_samples x (n_verts + 1) x 3
            

            for i_subd in range(opt.dataset.subdn - 1):
                # cur_subd_upsample = torch.zeros([num_samples, 0], dtype=torch.long)
                cur_subd_gt = torch.zeros([num_samples, 1, 3], dtype=torch.long).cuda()
                sample_batch[f'subd_{i_subd + 1}_gt'] = torch.cat(
                    [sample_batch[f'subd_{i_subd + 1}_gt'], cur_subd_gt], dim=1
                )
            
            nn_sampled += 1

            # subd_to_dist_xyz = self._create_dist_grid_coord_v2(subd_to_verts_embedding,  temperature=1., top_k=0, top_p=1.0, rt_logits=False)
            
            # loop_idx, cur_samples, outputs, cur_sample_probs, next_sample = self._loop_sample(loop_idx, cur_samples, embedding_layers=inputs_embedding_layers, project_to_logits_layer=project_to_logits, context_feature=context_feature, seq_context=seq_context, temperature=temperature, top_k=top_k, top_p=top_p)
            # # probs: bsz x n_cur_samples
            # samples = torch.cat(
            #     [samples, next_sample], dim=-1
            # )
            # probs = torch.cat(
            #     [probs, cur_sample_probs], dim=1
            # )

            if nn_sampled >= 213:
                break
        return sample_batch
    
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
            verts_dequantized = layer_utils.dequantize_verts(v, self.quantization_bits, min_range=opt.dataset.min_quant_range, max_range=opt.dataset.max_quant_range)
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


    def _sample_vertices_ndist(self, num_samples, context, context_feature): # sample #### context --> 
        # loop_idx = 0
        # samples = torch.zeros([num_samples, 0], dtype=torch.long).cuda()
        # probs = torch.ones([num_samples, 0], dtype=torch.float32).cuda()
        sample_batch = {}
        for i_subd in range(opt.dataset.subdn - 1):
            cur_subd_upsample = context[f'subd_{i_subd}'] # bsz x n_pts x 3
            # cur_subd_upsample = torch.zeros([num_samples, 0], dtype=torch.long)
            cur_subd_gt = torch.zeros([num_samples, 1, 3], dtype=torch.long).cuda()
            sample_batch[f'subd_{i_subd + 1}_gt'] = cur_subd_gt
            sample_batch[f'subd_{i_subd}'] = cur_subd_upsample
            
        nn_sampled = 0
        while True: # _loop_sample for samples
            subd_to_verts_embedding = self._embed_input(sample_batch, global_context_embedding=context_feature)

            subd_to_dist_xyz = self._create_dist_grid_coord_v2(context, subd_to_verts_embedding,  temperature=1., top_k=0, top_p=1.0, rt_logits=False)

            for i_subd in subd_to_dist_xyz:
                cur_sample = subd_to_dist_xyz[i_subd].sample() # bsz x n_verts x 3
                # cur_sample = cur_sample[:, -1:]
                sample_batch[f'subd_{i_subd + 1}_gt'][:, -1, 0] = cur_sample[:, -1, 0]
                # sample_batch[f'subd_{i_subd + 1}_gt'] = torch.cat([sample_batch[f'subd_{i_subd + 1}_gt'], cur_sample], dim=1) # num_samples x (n_verts + 1) x 3

            subd_to_verts_embedding = self._embed_input(sample_batch, global_context_embedding=context_feature)
            subd_to_dist_xyz = self._create_dist_grid_coord_v2(context, subd_to_verts_embedding,  temperature=1., top_k=0, top_p=1.0, rt_logits=False)

            for i_subd in subd_to_dist_xyz:
                cur_sample = subd_to_dist_xyz[i_subd].sample() # bsz x n_verts x 3
                # cur_sample = cur_sample[:, -1:]
                sample_batch[f'subd_{i_subd + 1}_gt'][:, -1, 1] = cur_sample[:, -1, 1]
                # sample_batch[f'subd_{i_subd + 1}_gt'] = torch.cat([sample_batch[f'subd_{i_subd + 1}_gt'], cur_sample], dim=1) # num_samples x (n_verts + 1) x 3

            
            subd_to_verts_embedding = self._embed_input(sample_batch, global_context_embedding=context_feature)
            subd_to_dist_xyz = self._create_dist_grid_coord_v2(context, subd_to_verts_embedding,  temperature=1., top_k=0, top_p=1.0, rt_logits=False)

            for i_subd in subd_to_dist_xyz:
                cur_sample = subd_to_dist_xyz[i_subd].sample() # bsz x n_verts x 3
                # cur_sample = cur_sample[:, -1:]
                sample_batch[f'subd_{i_subd + 1}_gt'][:, -1, 2] = cur_sample[:, -1, 2]
                # sample_batch[f'subd_{i_subd + 1}_gt'] = torch.cat([sample_batch[f'subd_{i_subd + 1}_gt'], cur_sample], dim=1) # num_samples x (n_verts + 1) x 3
            

            for i_subd in range(opt.dataset.subdn - 1):
                # cur_subd_upsample = torch.zeros([num_samples, 0], dtype=torch.long)
                cur_subd_gt = torch.zeros([num_samples, 1, 3], dtype=torch.long).cuda()
                sample_batch[f'subd_{i_subd + 1}_gt'] = torch.cat(
                    [sample_batch[f'subd_{i_subd + 1}_gt'], cur_subd_gt], dim=1
                )
            
            nn_sampled += 1

            # subd_to_dist_xyz = self._create_dist_grid_coord_v2(subd_to_verts_embedding,  temperature=1., top_k=0, top_p=1.0, rt_logits=False)
            
            # loop_idx, cur_samples, outputs, cur_sample_probs, next_sample = self._loop_sample(loop_idx, cur_samples, embedding_layers=inputs_embedding_layers, project_to_logits_layer=project_to_logits, context_feature=context_feature, seq_context=seq_context, temperature=temperature, top_k=top_k, top_p=top_p)
            # # probs: bsz x n_cur_samples
            # samples = torch.cat(
            #     [samples, next_sample], dim=-1
            # )
            # probs = torch.cat(
            #     [probs, cur_sample_probs], dim=1
            # )

            if nn_sampled >= 213:
                break
        return sample_batch
    


    def sample(self, num_samples, context=None, adapter_modules=None, max_sample_length=None, temperature=1., top_k=0, top_p=0.95, recenter_verts=True, only_return_complete=True, cond_context_info=False, sampling_max_num_grids=-1): # only the largets value is considered?
        # sample
        #### global context / seq context ####
        global_context, seq_context = self._prepare_context(context, adapter_module=adapter_modules)
        # global_contex
        # left_cond, rgt_cond = self._prepare_prediction(global_context)

        subdn = opt.dataset.subdn

        # for i_subd in range(subdn - 1):
        #   cur_subd_upsample = context['']

        # samples = tf.zeros([num_samples, 0], dtype=tf.int32)
        # samples = torch.zeros([num_samples, 0], dtype=torch.long)
        max_sample_length = max_sample_length
        self.max_sample_length = max_sample_length

        # sample vertice;  ######## 
        # loop_idx, samples, outputs, probs = self._sample_vertices(num_samples, inputs_embedding_layers=self.inputs_embedding_layers, project_to_logits=self.project_to_logits, context_feature=global_context, seq_context=seq_context, temperature=temperature, top_k=top_k, top_p=top_p)
        
        # max_samp
        cond_context = None if not cond_context_info else context
        sample_batch = self._sample_vertices(num_samples, context, global_context) # sample
        # loop_idx, samples = self._sample_grids(num_samples, global_context, seq_context, temperature, top_k, top_p, cond_context_info=cond_context_info, sample_context=cond_context, sampling_max_num_grids=sampling_max_num_grids)
        # grid_xyzs = samples['grid_xyzs'] - 1
        # grid_values = samples['grid_content_vocab']

        # outputs = {
        #   'grid_xyzs': grid_xyzs,
        #   'grid_values': grid_values
        # }
        outputs = sample_batch

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


