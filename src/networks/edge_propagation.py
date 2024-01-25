# from tkinter import N
import torch
import torch.nn as nn
from src.common_utils import data_utils_torch as data_utils


def edge_propagation(vert_features, edges, vert_dofs, feat_transform_layers):
  ### edges: bsz x 2 x n_edges ### edges_pair: (fr_idx, to_idx)
  ### vert_features: bsz x dim x N
  ### feat_transform_layers: composed of conv layers...
  fr_vert_idx = edges[:, 0, :] # - 1
  to_vert_idx = edges[:, 1, :] # - 1
  

  ### zero features for aggregation ###
  vert_features_aggr = torch.zeros_like(vert_features.contiguous().transpose(1, 2).contiguous())


  if vert_features.size(1) == 3: ### pts coordinates
    vert_features_aggr[:, fr_vert_idx] += vert_features.contiguous().transpose(1, 2).contiguous()[:, to_vert_idx] - vert_features.contiguous().transpose(1, 2).contiguous()[:, fr_vert_idx] ### vertices position differences 
    vert_features_aggr = vert_features_aggr / vert_dofs.unsqueeze(-1) ### avg of vertices coordinate differences
    vert_features = vert_features_aggr.contiguous().transpose(1, 2).contiguous()
  else:
    vert_features_aggr[:, fr_vert_idx] =  vert_features_aggr[:, fr_vert_idx] + vert_features.contiguous().transpose(1, 2).contiguous()[:, to_vert_idx]
    ### normlized feature aggregations ###
    vert_features_aggr = vert_features_aggr / vert_dofs.unsqueeze(-1)
    vert_features = vert_features + vert_features_aggr.contiguous().transpose(1, 2).contiguous()
  
  vert_features = feat_transform_layers(vert_features)
  return vert_features
  


class GraphConvNet(nn.Module):
  def __init__(self, n_layers=3):
    super().__init__()
    # self.tot_layers = nn.ModuleList()
    tot_conv_layers = []
    in_conv_layer = nn.Sequential(
      torch.nn.Conv1d(3, 128, 1),
      nn.BatchNorm1d(128),
      nn.Conv1d(128, 128, 1),
      nn.BatchNorm1d(128),
    )
    tot_conv_layers += [in_conv_layer]
    for i_layer in range(n_layers - 1):
      cur_conv_layer = [nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128),]
      if i_layer < n_layers - 2: ###
        cur_conv_layer.append(nn.ReLU())
      cur_conv_layer = nn.Sequential(*cur_conv_layer)
      tot_conv_layers.append(cur_conv_layer)
    self.tot_layers = nn.ModuleList(tot_conv_layers)
    
  def forward(self, vertices, edges, dofs):
    ## vertices, edges, dofs: list
    n_bsz = len(vertices)
    tot_out_features = []
    
    for i_bsz in range(n_bsz):
      cur_bsz_verts = vertices[i_bsz].unsqueeze(0)
      cur_bsz_verts = cur_bsz_verts.contiguous().transpose(1, 2).contiguous()
      cur_bsz_edges = edges[i_bsz]
      cur_bsz_dofs = dofs[i_bsz]
      for i_layer, cur_conv_layer in enumerate(self.tot_layers):
        # cur_bsz_verts = cur_conv_layer(cur_bsz_verts)
        cur_bsz_verts = edge_propagation(cur_bsz_verts, cur_bsz_edges, cur_bsz_dofs, cur_conv_layer)
      tot_out_features.append(cur_bsz_verts)
    return tot_out_features
  
def get_keypoints_features(src_keypts, src_vertices, src_tot_features):
  keypts_features = []
  n_bsz = len(src_vertices)
  for i_bsz in range(n_bsz):
    cur_bsz_keypts = src_keypts[i_bsz] ### n_kpts x 3
    cur_bsz_verts = src_vertices[i_bsz] ### n_verts x 3
    cur_bsz_dis_keypts_verts = torch.sum((cur_bsz_keypts.unsqueeze(1) - cur_bsz_verts.unsqueeze(0)) ** 2, dim=-1) ### n_kpts x n_verts
    _, cur_bsz_mapping = torch.min(cur_bsz_dis_keypts_verts, dim=-1)
    
    cur_bsz_features = src_tot_features[i_bsz].squeeze(0).contiguous().transpose(0, 1).contiguous()
    
    # print(f"cur_bsz_features: {cur_bsz_features.size()}, max_cur_bsz_mapping: {torch.max(cur_bsz_mapping)}, min_cur_bsz_mapping: {torch.min(cur_bsz_mapping)}")
    cur_bsz_kpts_features = data_utils.batched_index_select(values=cur_bsz_features, indices=cur_bsz_mapping, dim=0) ### n_keypoints x dim
    cur_bsz_kpts_features = data_utils.safe_transpose(cur_bsz_kpts_features, 0, 1)
    keypts_features.append(cur_bsz_kpts_features.unsqueeze(0))
  keypts_features = torch.cat(keypts_features, dim=0)
  # print(f"keypts_features: {keypts_features.size()}")
  return keypts_features
      
# def get_pc_features(pc, vertices, edges, dofs, c):


class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel, in_dim=3):
        super().__init__()
        self.encoder_channel = encoder_channel
        
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_dim, 128, 1), ### coordiantes conv 
            nn.BatchNorm1d(128), ### 
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , feat_dim = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, feat_dim)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n; 
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


  
class KNNFeatureBlock(nn.Module):
  def __init__(self, k=512, encode_feat_dim=128, in_dim=3):
    super().__init__()
    self.k = k
    self.encode_feat_dim = encode_feat_dim
    self.pc_block_encoder = Encoder(self.encode_feat_dim, in_dim=in_dim)
  
  def forward(self, pos, feats=None): ### jsut pose is enough..
    ### pos: bsz x N x 3
    ### bsz x N x 1 x 3 - bsz x 1 x N x 3 ---> bsz x N x N x 3
    ppdist = torch.sum((pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2, dim=-1)
    ### ppdist: sqrt distance ###
    ppdist = torch.sqrt(ppdist)
    topk_dist, topk_idx = torch.topk(ppdist, k=self.k, dim=2, largest=False) #### topk_idx: bsz x n_samples --> sampled topk points' indexes
    
    grouped_pos = data_utils.batched_index_select(values=pos, indices=topk_idx, dim=1)
    
    ### grouped_pose: bsz x N x K x 3
    ### batched_index_select --> for batch indexes
    grouped_pos = grouped_pos - pos.unsqueeze(2) ### bsz x N x K x 3
    # grouped_pos = grouped_po
    
    if feats is not None:
      # print(f"feats: {feats.size()}")
      grouped_feats = data_utils.batched_index_select(values=feats, indices=topk_idx, dim=1)
      grouped_pos = torch.cat([grouped_pos, grouped_feats], dim=-1)
      # print(f"grouped_pos: {grouped_pos.size()}")
    
    grouped_feat = self.pc_block_encoder(grouped_pos) ### bsz x N x encode_feat_dim
    grouped_feat = grouped_feat.contiguous().transpose(1, 2).contiguous()
    return grouped_feat, topk_idx
    
    