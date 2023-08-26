import torch
import  numpy as np

def dequantize_verts(verts, n_bits=8, add_noise=False):
  """Convert quantized vertices to floats."""
  min_range = -0.5
  max_range = 0.5
  range_quantize = 2**n_bits - 1
  # verts = verts.astype('float32')
  verts = verts.float()
  verts = verts * (max_range - min_range) / range_quantize + min_range
  # if add_noise:
  #   verts += np.random.uniform(size=verts.shape) * (1 / range_quantize)
  return verts


def quantize_verts(verts, n_bits=8):
  """Convert vertices in [-1., 1.] to discrete values in [0, n_bits**2 - 1]."""
  min_range = -0.5
  max_range = 0.5
  range_quantize = 2**n_bits - 1
  verts_quantize = (verts - min_range) * range_quantize / (
      max_range - min_range)
  return verts_quantize.long()

def embedding_to_padding(x):
    # x: bz x seq_len x emb_dim
    x_sum = torch.sum(torch.abs(x), dim=-1, keepdim=False) # abs to zeors..
    x_padding_indicators = (x_sum < 1e-9).float() # L x 
    # x_padding_indicators: bz x seq_len
    # (bz x 1 x seq_len) + (bz x seq_len x 1) --> (bz x seq_len x seq_len) --- and the zeros
    
    return x_padding_indicators

def attention_bias_ignore_padding(x):
    negative_value = -1e9
    negative_value = torch.full_like(x, fill_value=negative_value) 

    ret = x * negative_value
    ret = ret.unsqueeze(1).unsqueeze(1)
    return ret

def attention_mask(x):
    mask = x.unsqueeze(1) + x.unsqueeze(-1)
    mask = (mask > 0.5)
    return mask

def attention_mask_single_direction(x, other_len=None):
    if other_len is None:
        x_other_dir = torch.zeros_like(x)
    else:
        x_other_dir = torch.zeros((x.size(0), other_len), dtype=torch.float32).cuda()
    mask = x_other_dir.unsqueeze(-1) + x.unsqueeze(1) # bz x seq_len_query x seq_len
    mask = (mask > 0.5)
    return mask

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

def top_k_logits(logits, k):
  """Masks logits such that logits not in top-k are small."""
  if k == 0:
    return logits
  else:
    # values, _ = tf.math.top_k(logits, k=k)
    values, _ = torch.topk(logits, dim=-1, k=k)
    k_largest = torch.min(values, dim=-1)
    # k_largest = tf.reduce_min(values)
    # if the logit value is not bigger than the k-th largest value...
    logits = torch.where(logits <= k_largest, torch.ones_like(logits) * (-1e9), logits)
    # logits = tf.where(tf.less_equal(logits, k_largest),
    #                   tf.ones_like(logits)*-1e9, logits)
    return logits

# 
# disable high sampling probailities?
# to prevent over-fitting?
def top_p_logits(logits, p, testing=False):
    # p = 1.0 if testing is False else p
    # p = 1.0s
    if p == 1:
        return logits
    else:
        # logit_shape = tf.shape(logits)
        seq, dim = logits.size(1), logits.size(2)
        # seq, dim = logit_shape[1], logit_shape[2]
        # logits = tf.reshape(logits, [-1, dim])
        logits = logits.contiguous().view(-1, dim).contiguous()
        sort_indices = torch.argsort(logits, dim=-1, descending=True) # (bz x seq) x n_dim_logits
        reverse_sort_indices = torch.argsort(sort_indices, dim=-1, descending=False)
        # sort_indices = tf.argsort(logits, axis=-1, direction='DESCENDING')
        # 
        # probs = torch.gather(torch.softmax(logits, dim=-1), dim=1, index=sort_indices)
        # gather sorted probabilities
        probs = batched_index_select(values=torch.softmax(logits, dim=-1), indices=sort_indices, dim=1)
        
        # probs

        # probs = tf.gather(tf.nn.softmax(logits), sort_indices, batch_dims=1)  #
        # cumprobs = tf.cumsum(probs, axis=-1, exclusive=True)
        cumprobs = torch.cumsum(probs, dim=-1)
        # The top 1 candidate always will not be masked.
        # This way ensures at least 1 indices will be selected.
        # print("cumprobs,", cumprobs)
        # sort_mask = (cumprobs > p).float() # cumprobs...
        sort_mask = (cumprobs > p)
        # print(sort_mask)
        # sort_mask = ()
        # top_p_mask = batched_index_select(values=sort_mask, indices=sort_indices, dim=1)
        top_p_mask = batched_index_select(values=sort_mask, indices=reverse_sort_indices, dim=1)
        # batch_indices = tf.tile(
        #     tf.expand_dims(tf.range(tf.shape(logits)[0]), axis=-1), [1, dim])
        # top_p_mask = tf.scatter_nd(
        #     tf.stack([batch_indices, sort_indices], axis=-1), sort_mask,
        #     tf.shape(logits))
        # logits -= top_p_mask * 1e9
        logits[top_p_mask] = -1e9
        # print(logits)s
        logits = logits.contiguous().view(-1, seq, dim).contiguous()
        return logits
        # return tf.reshape(logits, [-1, seq, dim])  #