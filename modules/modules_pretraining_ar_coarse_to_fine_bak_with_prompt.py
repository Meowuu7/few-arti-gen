# import sonnet as snt
# from tensor2tensor.layers import common_attention
# from tensor2tensor.layers import common_layers
# import tensorflow.compat.v1 as tf
# from tensorflow.python.framework import function
# import tensorflow_probability as tfp

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
        self.num_heads = 1
        self.layer_norm = layer_norm
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.re_zero = re_zero
        self.memory_efficient = memory_efficient
        self.category_part_indicator = opt.dataset.category_part_indicator

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

    def forward(self, inputs, inputs_mask=None,  adapter_module=None):
        ### padding 
        if inputs_mask is None:
            encoder_padding = layer_utils.embedding_to_padding(inputs) # bsz x n_vertices
        else:
            encoder_padding = inputs_mask # inputs_mask: bsz x n_vertices
        bsz = inputs.size(0)
        seq_length = inputs.size(1)

        prompt_inputs = adapter_module.prefix_keys[0].contiguous().repeat(bsz, 1, 1).contiguous()
        # inputs: bsz x (seq_lenght + prefix_length) x embedding_dim
        inputs = torch.cat([prompt_inputs, inputs], dim=1)
        # 
        prompt_inputs_length = adapter_module.prefix_keys[0].size(1)
        prompt_inputs_mask = torch.zeros((bsz, prompt_inputs_length), dtype=torch.float32).cuda()
        # encoder_padding: encoder_padding...
        encoder_padding = torch.cat([prompt_inputs_mask, encoder_padding], dim=1) # bsz x (seq_length + prefix_length)

        # inputs: bsz x seq_length x embedding_dim
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
        if self.category_part_indicator:
          x = torch.cat(
            [x[:, prompt_inputs_length - 1: prompt_inputs_length, :], x[:, prompt_inputs_length + 1: , :]], dim=1
          )
        else:
          x = x[:, prompt_inputs_length:, :]
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
        self.category_part_indicator = opt.dataset.category_part_indicator

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

    def forward(self, inputs, sequential_context_embeddings=None, mask=None, sequential_context_mask=None, adapter_module=None):
        seq_length = inputs.size(1)
        bsz = inputs.size(0)

        prompt_inputs = adapter_module.prefix_keys[0].contiguous().repeat(bsz, 1, 1).contiguous()
        prompt_inputs_length = adapter_module.prefix_keys[0].size(1)
        # inputs: bsz x (seq_lenght + prefix_length) x embedding_dim
        inputs = torch.cat([prompt_inputs, inputs], dim=1)

        seq_length = inputs.size(1)
        bsz = inputs.size(0)

        atten_mask = np.tri(seq_length, seq_length, -1.0, dtype=np.float32).T # tri
        # atten_mask = np.tri(seq_length, seq_length, 0.0, dtype=np.float32)
        atten_mask = torch.from_numpy(atten_mask).float().cuda()
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
        # x = x[:, prompt_inputs_length:, :]
        if self.category_part_indicator:
          if self.debug:
            print("here in catting")
          x = torch.cat(
            [x[:, prompt_inputs_length - 1: prompt_inputs_length, :], x[:, prompt_inputs_length + 1: , :]], dim=1
          )
        else:
          x = x[:, prompt_inputs_length:, :]
        return x


class VertexModel(nn.Module): # vertex model part?
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
                 ):
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

        # construct encoders and decoders # decoder; 
        self.decoder = TransformerDecoder(**decoder_config)

        # class embedding
        self.class_embedding_layer = nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.embedding_dim)
        # torch.nn.init(files=None)
        torch.nn.init.xavier_uniform_(self.class_embedding_layer.weight)

        self.inputs_embedding_layers = nn.ModuleList([
            nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim), # tf.mod(tf.range(seq_length), 3)
            nn.Embedding(num_embeddings=self.max_num_input_verts + 5, embedding_dim=self.embedding_dim), # tf.floordiv(tf.range(seq_length), 3)
            nn.Embedding(num_embeddings=2**self.quantization_bits + 2, embedding_dim=self.embedding_dim) # quantized vertices
        ])
        for cur_layer in self.inputs_embedding_layers:
            torch.nn.init.xavier_uniform_(cur_layer.weight)

        if not self.class_conditional: # class condition # claass condition
            self.zero_embed = nn.Parameter(torch.zeros(size=(1, 1, self.embedding_dim), requires_grad=True, dtype=torch.float32), requires_grad=True)


        # logits prediction, joint dir and joint pvp prediction
        # self.project_to_pointer_inter_part = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.project_to_logits = nn.Linear(self.embedding_dim, 2**self.quantization_bits + 1, bias=True)
        torch.nn.init.xavier_uniform_(self.project_to_logits.weight)

    def _prepare_context(self, context):
        if self.class_conditional:
            global_context_embedding = self.class_embedding_layer(context['class_label'])
        else:
            global_context_embedding = None
        return  global_context_embedding, None

    def _predict_joint(self, cond_feature):
        joint_dir = self.joint_dir_pred_layer(cond_feature)
        joint_pvp = self.joint_pvp_pred_layer(cond_feature)
        joint_dir = joint_dir / torch.clamp(torch.norm(joint_dir, p=2, dim=-1, keepdim=True), min=1e-9)
        return joint_dir, joint_pvp

    # calcualte input vertices embedding given the embedding layer, context embedding, and vertices values
    def _embed_inputs(self, vertices, embedding_layers, global_context_embedding=None):
        bz = vertices.size(0)
        # check vertices values
        # print(f"maxx of vertices: {torch.max(vertices, dim=-1)[0]}, minn of vertices: {torch.min(vertices, dim=-1)}")
        seq_length = vertices.size(1)
        vertices_coords = torch.arange(0, seq_length).cuda() % 3
        vertices_coords_embedding = embedding_layers[0](vertices_coords)
        vertices_pos = torch.floor_divide(torch.arange(0, seq_length).cuda(), 3) # vertex index
        vertices_pos_embedding = embedding_layers[1](vertices_pos)
        vertices_embedding = embedding_layers[2](vertices)
        vertices_embedding = (vertices_coords_embedding + vertices_pos_embedding).unsqueeze(0) + vertices_embedding
        if global_context_embedding is None: # use zero_embed for global context embedding... # global context
            global_context_embedding = self.zero_embed.repeat(bz, 1, 1)
        #### embed input here ####
        # print(f"_embed_inputs, global_context_embedding: {global_context_embedding.size()}, vertex_embedding: {vertices_embedding.size()}")
        if len(global_context_embedding.size()) == 2:
            global_context_embedding = global_context_embedding.unsqueeze(1)
        vertices_embedding = torch.cat([global_context_embedding, vertices_embedding], dim=1)
        # if self.training:
        #     print(f"vertices embeddings: {vertices_embedding.size()}")
        return vertices_embedding

    def _create_dist(self, vertices, embedding_layers, project_to_logits_layer, global_context_embedding=None, sequential_context_embeddings=None, temperature=1., top_k=0, top_p=1.0, adapter_module=None):
        # todo: cache!
        decoder_inputs = self._embed_inputs(vertices, embedding_layers, global_context_embedding=global_context_embedding)
        
        # decoder_inputs = decoder_inputs * vertices_mask.unsqueeze(-1)
        
        # 
        outputs = self.decoder(decoder_inputs, adapter_module=adapter_module)
        logits = project_to_logits_layer(outputs)
        # print("current logits, ", logits)
        logits /= temperature
        # logits
        logits = layer_utils.top_k_logits(logits, top_k)
        # logits = layer_utils.top_p_logits(logits, top_p, testing=not self.training)
        logits = layer_utils.top_p_logits(logits, top_p)

        cat_dist = torch.distributions.Categorical(logits=logits)
        # cat_dist = tfd.Categorical(logits=logits)
        return cat_dist, outputs

    def forward(self, batch, return_global_feature=False, adapter_module=None):
        if 'global_context' in batch:
          global_context = batch['global_context']
          seq_context = None
        else:
          global_context, seq_context = self._prepare_context(batch)
        # left_cond, rgt_cond = self._prepare_prediction(global_context) # _create_dist() #
        # project to logits layer 
        pred_dist, outputs = self._create_dist(batch['vertices_flat'][:, :-1], embedding_layers=self.inputs_embedding_layers, project_to_logits_layer=self.project_to_logits, global_context_embedding=global_context, sequential_context_embeddings=seq_context, adapter_module=adapter_module)
        
        if return_global_feature:
          num_vertices = batch['num_vertices'] # bsz
          # conditioned on features from all points
          global_feature = layer_utils.batched_index_select(values=outputs, indices=num_vertices.long().unsqueeze(1), dim=1).squeeze(1)
          return pred_dist, global_feature
        else:
          return pred_dist

    def _loop_sample(self, loop_idx, samples, embedding_layers, project_to_logits_layer, context_feature, seq_context, top_k, top_p, adapter_module=None):
        cat_dist, outputs = self._create_dist(
            samples, embedding_layers=embedding_layers, project_to_logits_layer=project_to_logits_layer, global_context_embedding=context_feature, sequential_context_embeddings=seq_context, top_k=top_k, top_p=top_p, adapter_module=adapter_module
        )
        next_sample = cat_dist.sample()
        next_sample = next_sample[:, -1:] # next
        samples = torch.cat([samples, next_sample], dim=1)
        # print(f"in vertex looping sampling, samples: {samples.size()}")
        return loop_idx + 1, samples, outputs

    def _stop_cond(self, samples):
        # print(f"in stop condition, samples: {samples}")
        equal_to_zero = (samples == 0).long() # equal to zero

        accum_equal_zero = torch.sum(equal_to_zero, dim=-1)
        not_stop = (accum_equal_zero == 0).long() # 
        not_stop = torch.sum(not_stop).item() > 0 # not stop
        return (not not_stop)

    def _sample_vertices(self, num_samples, inputs_embedding_layers, project_to_logits, context_feature, seq_context, top_k, top_p, adapter_module=None): # sample
        loop_idx = 0
        # init samples
        samples = torch.zeros([num_samples, 0], dtype=torch.long).cuda()
        while True: # _loop_sample for samples
            loop_idx, samples, outputs = self._loop_sample(loop_idx, samples, embedding_layers=inputs_embedding_layers, project_to_logits_layer=project_to_logits, context_feature=context_feature, seq_context=seq_context, top_k=top_k, top_p=top_p, adapter_module=adapter_module)
            if self._stop_cond(samples) or loop_idx >= self.max_sample_length * 3:
                break
        return loop_idx, samples, outputs

    def sample(self, num_samples, context=None, max_sample_length=None, temperature=1., top_k=0, top_p=0.95, recenter_verts=True, only_return_complete=True, adapter_module=None): # only the largets value is considered?
        # sample
        if 'global_context' in context:
          global_context = context['global_context']
          seq_context = None
        else:
          global_context, seq_context = self._prepare_context(context)
        # global_contex
        # left_cond, rgt_cond = self._prepare_prediction(global_context)

        # samples = tf.zeros([num_samples, 0], dtype=tf.int32)
        # samples = torch.zeros([num_samples, 0], dtype=torch.long)
        max_sample_length = max_sample_length
        self.max_sample_length = max_sample_length

        # sample vertice; 
        loop_idx, samples, outputs = self._sample_vertices(num_samples, inputs_embedding_layers=self.inputs_embedding_layers, project_to_logits=self.project_to_logits, context_feature=global_context, seq_context=seq_context, top_k=top_k, top_p=top_p, adapter_module=adapter_module)

        # print(f"ori sampled vertex size: {samples.size()}")
        completed = torch.any(samples == 0, dim=-1) 
        stop_index_completed = torch.argmax((samples == 0).long(), dim=-1)
        stop_index_incomplete = (max_sample_length * 3 * torch.ones_like(stop_index_completed))
        # select the stop indexes
        stop_index = torch.where(completed, stop_index_completed, stop_index_incomplete) # stop index
        num_vertices = torch.floor_divide(stop_index, 3) # 

        num_sampled_vertices_len = torch.minimum(num_vertices * 3, torch.tensor([outputs.size(1) - 1], dtype=torch.long).cuda())
        vertices_global_feature = data_utils.batched_index_select(values=outputs, indices=num_sampled_vertices_len.unsqueeze(-1), dim=1).squeeze(1) # vertices_global_feature: bsz x num_features_dim

        print(f"number of vertices: {num_vertices}") # number of vertices for all samples

        # print(f"vertex samples size: {samples.size()}")

        v = samples
        v = v[:, :(torch.max(num_vertices) * 3)] - 1
        verts_dequantized = layer_utils.dequantize_verts(v, self.quantization_bits)
        # vertices
        vertices = verts_dequantized.contiguous().view(num_samples, -1, 3).contiguous()
        # z, x, y --> y, x, z?
        vertices = torch.cat([vertices[..., 2].unsqueeze(-1), vertices[..., 1].unsqueeze(-1), vertices[..., 0].unsqueeze(-1)], dim=-1)

        # vertices; 
        if max_sample_length > vertices.size(1):
            pad_size = max_sample_length - vertices.size(1)
            pad_zeros = torch.zeros((num_samples, pad_size, 3), dtype=torch.float32).cuda()
            # padding
            vertices = torch.cat([vertices, pad_zeros], dim=1)
        else:
            vertices = vertices[:, :max_sample_length]

        vertices_mask = (torch.arange(0, max_sample_length).unsqueeze(0).cuda() < num_vertices.unsqueeze(1)).float() # valid ones

        ### centralize vertices ### # and use the vertices 
        if recenter_verts:
            vert_max, _ = torch.max( # max pooling for the maximum vertices value
                vertices - 1e10 * (1. - vertices_mask).unsqueeze(-1), dim=1, keepdim=True)
            vert_min, _ = torch.min( # min pooling for the minimum vertices value
                vertices + 1e10 * (1. - vertices_mask).unsqueeze(-1), dim=1, keepdim=True)
            vert_centers = 0.5 * (vert_max + vert_min) # centers
            vertices -= vert_centers # centralize vertices # vertices
        vertices *= vertices_mask.unsqueeze(-1)# vertices mask?

        outputs = {
            'completed': completed,  #
            'vertices': vertices,  # dequantized vertices
            'num_vertices': num_vertices,
            'vertices_mask': vertices_mask,
            'vertices_global_feature': vertices_global_feature # bsz x num_feature_dim
        }

        return outputs


class FaceModel(nn.Module):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 class_conditional=False,
                 num_classes=55,
                 decoder_cross_attention=True, # cross attention
                 use_discrete_vertex_embeddings=True,
                 quantization_bits=8,
                 max_seq_length=5000,
                 ):
        super(FaceModel, self).__init__()
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.decoder_cross_attention = decoder_cross_attention
        self.use_discrete_vertex_embeddings = use_discrete_vertex_embeddings
        self.quantization_bits = quantization_bits
        self.max_seq_length = max_seq_length
        self.embedding_dim = decoder_config['hidden_size']

        ### Decoders and Encoders using Transformers
        self.decoder = TransformerDecoder(**decoder_config)
        self.encoder = TransformerEncoder(**encoder_config)
        # self.decoder_rgt = TransformerDecoder(**decoder_config)
        # self.encoder_rgt = TransformerEncoder(**encoder_config)

        # class embedding
        self.class_embedding_layer = nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.embedding_dim)

        # quantized vertex embedding layer
        # self.vertex_embedding_layer = nn.Embedding(num_embeddings=2**self.quantization_bits + 1, embedding_dim=self.embedding_dim)
        # for xyz embedding and stopping embedding var
        self.vertex_embedding_layer = nn.ModuleList(
            [nn.Embedding(num_embeddings=2 ** self.quantization_bits + 1, embedding_dim=self.embedding_dim),
              nn.Embedding(num_embeddings=2 ** self.quantization_bits + 1, embedding_dim=self.embedding_dim),
              nn.Embedding(num_embeddings=2 ** self.quantization_bits + 1, embedding_dim=self.embedding_dim)
              ]
        )
        self.vertex_stopping_embedding_var = nn.Parameter(torch.zeros((1, 2, self.embedding_dim), dtype=torch.float32, requires_grad=True), requires_grad=True)
        
        self.face_vertices_embedding_layer = nn.Embedding(self.max_seq_length + 5, self.embedding_dim)
        self.face_vertices_zero_embedding_var = nn.Parameter(torch.zeros((1, 1, self.embedding_dim), dtype=torch.float32, requires_grad=True), requires_grad=True)

        # self.face_vertices_embedding_layer_rgt = nn.Embedding(self.max_seq_length + 5, self.embedding_dim)
        # self.face_vertices_zero_embedding_var_rgt = nn.Parameter(
        #     torch.zeros((1, 1, self.embedding_dim), dtype=torch.float32, requires_grad=True), requires_grad=True)

        self.pointer_project_layer = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        # self.pointer_project_layer_rgt = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)


    def _embed_class_label(self, labels):
        class_embedding = self.class_embedding_layer(labels)
        return class_embedding


    def _embed_vertices(self, vertices, vertices_mask, vertex_embedding_layer, vertex_stopping_embedding_var, encoder_adapter=None):
        # vertex embedding layer, stopping embedding var
        # embed vertices
        vertex_embeddings = 0.
        
        if not self.training:
            verts_quantized = layer_utils.quantize_verts(vertices, self.quantization_bits)
        else:
            verts_quantized = vertices
        for c in range(3):
            curr_coordinate_embeddings = vertex_embedding_layer[c](verts_quantized[..., c])
            vertex_embeddings += curr_coordinate_embeddings

        vertex_embeddings *= vertices_mask.unsqueeze(-1)
        stopping_embeddings = vertex_stopping_embedding_var.contiguous().repeat(vertices.size(0), 1, 1).contiguous()
        vertex_embeddings = torch.cat(
            [stopping_embeddings, vertex_embeddings], dim=1
        )
        # 
        vertices_mask_for_encoder = torch.cat(
            [torch.ones((vertices_mask.size(0), 2), dtype=torch.float32).cuda(), vertices_mask], dim=-1
        )
        vertices_mask_for_encoder = 1.0 - vertices_mask_for_encoder # not 
        # print(f"after adding stopping embedding, vertex_embeddings: {vertex_embeddings.size()}")
        
        vertex_embeddings = self.encoder(vertex_embeddings, vertices_mask_for_encoder, adapter_module=encoder_adapter)
        # print(f"After vertex encoder! vertex_embeddings: {vertex_embeddings.size()}")
        return vertex_embeddings

    def _prepare_context(self, context, encoder_adapter=None):
        if self.class_conditional: # class_label
            global_context_embedding = self._embed_class_label(context['class_label'])
        else:
            global_context_embedding = None
        # 
        vertex_embeddings = self._embed_vertices(
            context['vertices'],
            context['vertices_mask'], self.vertex_embedding_layer, self.vertex_stopping_embedding_var, 
            encoder_adapter=encoder_adapter)
        
        if self.decoder_cross_attention:
            # vertex_embedding --- bz x n_seq x dim
            vertices_mask_padding = context['vertices_mask']
            # vertices_mask_padding = torch.cat(
            #     [vertices_mask_padding, torch.zeros((vertex_embeddings.size(0), 2), dtype=torch.float32).cuda()], dim=-1
            # )
            vertices_mask_padding = torch.cat(
                [torch.ones((vertex_embeddings.size(0), 2), dtype=torch.float32).cuda(), vertices_mask_padding], dim=-1
            )
            sequential_context_embeddings = vertex_embeddings * vertices_mask_padding.unsqueeze(-1)
        else:
            sequential_context_embeddings = None
        return (vertex_embeddings, global_context_embedding, sequential_context_embeddings)


    def _embed_inputs(self, faces_long, vertex_embeddings, face_vertices_embedding_layer, face_vertices_zero_embedding_var, global_context_embedding=None):
        """Embeds face sequences and adds within and between face positions."""
        # try:
        #     print(f"vertex_embeddings: {vertex_embeddings.size()}, max_of_faces_long: {torch.max(faces_long, dim=-1)[0]}, min_of_faces_long: {torch.min(faces_long, dim=-1)[0]}")
        # except:
        #     pass
        # vertex_embeddings: bz x [2 + N_v] x dim
        # faces_long: --- it should not be the second or the first embedding? --- so faces long?
        if torch.any(torch.isnan(vertex_embeddings)):
            raise ValueError("NaN value in vertex_embeddings (face_model._embed_inputs)...")
        face_embeddings = data_utils.batched_index_select(values=vertex_embeddings, indices=faces_long, dim=1)
        # print(f"in embd inputs with faces_embedding: {face_embeddings.size()}")
        pos_embeddings = face_vertices_embedding_layer(torch.arange(faces_long.size(1)).cuda())
        # zero_embedding_tiles for zero_embedding_layer
        zero_embedding_tiles = face_vertices_zero_embedding_var.repeat(faces_long.size(0), 1, 1)
        # face embeddings -- face embeddings for vertices coordinates + pos embedding for vertices indices
        embeddings = face_embeddings + pos_embeddings.unsqueeze(0)
        # get embeddings
        embeddings = torch.cat([zero_embedding_tiles, embeddings], dim=1) # zero embedding for the first face index prediction? # embedd
        # the order of the faces --- the first face's first vertex and ...
        return embeddings

    def _project_to_pointers(self, inputs, project_to_pointers_layer):
        projected_pointers = project_to_pointers_layer(inputs)
        return projected_pointers

    def _create_dist(self,
                     vertex_embeddings,
                     vertices_mask,
                     faces_long,
                     face_vertices_embedding_layer,
                     face_vertices_zero_embedding_var,
                     decoder_module,
                     project_to_pointers_layer,
                     global_context_embedding=None,
                     sequential_context_embeddings=None,
                     temperature=1.,
                     top_k=0,
                     top_p=1.0,
                     is_training=False,
                     cache=None,
                     rt_decoder_output=False,
                     decoder_adapter=None): # and we should use other strategies for finetuning...
        """Outputs categorical dist for vertex indices."""
        # Embed inputs
        # face vertex embeddings with the stopping index and the new index?
        decoder_inputs = self._embed_inputs(
            faces_long, vertex_embeddings, face_vertices_embedding_layer, face_vertices_zero_embedding_var, global_context_embedding)

        sequential_context_mask = torch.cat((torch.ones((vertices_mask.size(0), 2), dtype=torch.float32).cuda(), vertices_mask), dim=-1)

        if torch.any(torch.isnan(decoder_inputs)):
            # print(f"NaN in decoder_inputs of the face model while sampling...")
            raise ValueError(f"NaN in decoder_inputs of the face model while sampling...")
        decoder_outputs = decoder_module(
            decoder_inputs,
            sequential_context_embeddings=sequential_context_embeddings,
            sequential_context_mask=sequential_context_mask, adapter_module=decoder_adapter
            )

        if not self.training:
            decoder_outputs = decoder_outputs[:, -1:]
        # Get pointers
        # decoder outputs --- to pointers # current predicted pointers
        # [first vertex, ] # pointers
        pred_pointers = self._project_to_pointers(decoder_outputs, project_to_pointers_layer)

        logits = torch.matmul(pred_pointers, vertex_embeddings.contiguous().transpose(1, 2).contiguous())
        # self embedding dim
        logits = logits / math.sqrt(float(self.embedding_dim))
        # f_verts_mask = tf.pad(
        #     vertices_mask, [[0, 0], [2, 0]], constant_values=1.)[:, None]
        # logits *= f_verts_mask  #
        # logits -= (1. - f_verts_mask) * 1e9
        # face verts mask
        f_verts_mask = torch.cat([torch.ones((logits.size(0), 2), dtype=torch.float32).cuda(), vertices_mask], dim=-1).unsqueeze(1)
        # print(f"In face model creat_dist, f_verts_mask: {f_verts_mask.size()}, logits: {logits.size()}")
        logits = logits * f_verts_mask
        logits -= (1. - f_verts_mask) * 1e9 # for logits
        logits /= temperature
        logits = layer_utils.top_k_logits(logits, top_k)
        logits = layer_utils.top_p_logits(logits, top_p)

        if rt_decoder_output:
            return torch.distributions.Categorical(logits=logits), decoder_outputs
        else:
            return torch.distributions.Categorical(logits=logits)

    def forward(self, batch, rt_global_features=False, encoder_adapter=None, decoder_adapter=None):
        
        vertex_embeddings, global_context, seq_context = self._prepare_context(batch, encoder_adapter=encoder_adapter)
        # vertex_embeddings_rgt, global_context_rgt, seq_context_rgt = self._prepare_context_rgt(batch)
        # vertex embeddings
        
        
        
        pred_dist, decoder_output = self._create_dist(
            vertex_embeddings, batch['vertices_mask'], batch['faces'][:, :-1], face_vertices_embedding_layer=self.face_vertices_embedding_layer, face_vertices_zero_embedding_var=self.face_vertices_zero_embedding_var, decoder_module=self.decoder, project_to_pointers_layer=self.pointer_project_layer, global_context_embedding=global_context, sequential_context_embeddings=seq_context, rt_decoder_output=True,  decoder_adapter=decoder_adapter
        )

        num_faces = batch['num_faces'] # bsz
        # faces_embeddings: bsz x feature_dim
        faces_embeddings = data_utils.batched_index_select(values=decoder_output, indices=num_faces.unsqueeze(-1), dim=1).squeeze(1)
        
        if rt_global_features:
            return pred_dist, faces_embeddings
        else:
            return pred_dist
        
        # return pred_dist, 


    def _loop_sample(self, loop_idx, samples, vertex_embeddings, vertices_mask, face_vertices_embedding_layer, face_vertices_zero_embedding_var, decoder_module, project_to_pointers_layer, context_feature, seq_context, top_k, top_p, decoder_adapter=None):
        # face 
        cat_dist = self._create_dist(vertex_embeddings, vertices_mask, samples, face_vertices_embedding_layer, face_vertices_zero_embedding_var, decoder_module, project_to_pointers_layer, context_feature, seq_context, top_k=top_k, top_p=top_p, decoder_adapter=decoder_adapter)

        next_sample = cat_dist.sample()

        # vertices_mask
        # print(next_sample)

        samples = torch.cat([samples, next_sample], dim=1)
        return loop_idx + 1, samples

    def _stop_cond(self, samples):
        # todo: check this!
        equal_to_zero = (samples == 0).long() # bz x seq
        accum_equal_zero = torch.sum(equal_to_zero, dim=-1) # bz ---
        not_stop = (accum_equal_zero == 0).long()
        not_stop = torch.sum(not_stop).item() > 0 # at least one sample should not be stopped
        return (not not_stop)

    # _sample_vertices
    def _sample_vertices(self, num_samples, vertex_embeddings, vertices_mask, face_vertices_embedding_layer, face_vertices_zero_embedding_var, decoder_module, project_to_pointers_layer, context_feature, seq_context, max_sample_length, top_k, top_p, decoder_adapter=None):
        loop_idx = 0
        samples = torch.zeros([num_samples, 0], dtype=torch.long).cuda()
        while True:
            loop_idx, samples = self._loop_sample(loop_idx, samples, vertex_embeddings, vertices_mask, face_vertices_embedding_layer, face_vertices_zero_embedding_var, decoder_module, project_to_pointers_layer, context_feature, seq_context, top_k=top_k, top_p=top_p,
            decoder_adapter=decoder_adapter)
            if self._stop_cond(samples) or loop_idx >= max_sample_length:
                break
        return loop_idx, samples

    def sample(self,
               context,
               max_sample_length=None,
               temperature=1., # temperature
               top_k=0, # top_k
               top_p=0.95, # top_p
               only_return_complete=True,
               encoder_adapter=None,
               decoder_adapter=None):
        # vertex embeddings, global
        vertex_embeddings, global_context, seq_context = self._prepare_context(context, encoder_adapter=encoder_adapter)
        num_samples = vertex_embeddings.size(0)

        #### max sample length --- max sampling length; max sequence length
        max_sample_length = max_sample_length # or self.max_seq_length

        loop_idx, f = self._sample_vertices(
            num_samples, vertex_embeddings, context['vertices_mask'], self.face_vertices_embedding_layer, self.face_vertices_zero_embedding_var, self.decoder, self.pointer_project_layer, global_context, seq_context, max_sample_length, top_k=top_k, top_p=top_p, decoder_adapter=decoder_adapter
        )

        complete_samples = torch.any((f == 0).float(), axis=-1)

        sample_length = f.size(-1)
        max_one_ind = torch.max(torch.arange(sample_length).unsqueeze(0).cuda() * (f == 1).long(), dim=-1)[0]
        zero_inds = (torch.argmax((f == 0).long(), dim=-1)).long()
        num_face_indices = torch.where(complete_samples, zero_inds, max_one_ind) + 1

        # Mask faces beyond stopping token with zeros
        faces_mask = (torch.arange(0, sample_length).unsqueeze(0).cuda() < num_face_indices.unsqueeze(1) - 1).long()
        f *= faces_mask
        # faces_mask = (torch.arange(0, sample_length).unsqueeze(0) < num_face_indices.unsqueeze(1)).long()

        # Pad to maximum size with zeros
        if max_sample_length > sample_length: # sa
            pad_size = max_sample_length - sample_length
            f = torch.cat([f, torch.zeros((num_samples, pad_size), dtype=torch.long).cuda()], dim=-1)
        else:
            f = f[:, :max_sample_length]

        # outputs
        outputs = {
            # 'context': context['left'],
            'completed': complete_samples,
            'faces': f,
            'num_face_indices': num_face_indices,
        }
        return outputs


