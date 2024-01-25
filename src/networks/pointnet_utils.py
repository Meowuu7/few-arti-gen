import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn as nn

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1) ##
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64) 
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x    
    
class pointnet_encoder(nn.Module):
    def __init__(self):
        super(pointnet_encoder, self).__init__()
        self.channel = 3
        self.stn = STN3d(self.channel)
        self.conv1 = torch.nn.Conv1d(self.channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fstn = STNkd(k = 128)

    def forward(self, point_cloud, return_global):
        # point_cloud = point_cloud 
        point_cloud = point_cloud.transpose(2, 1)
        B, D, N = point_cloud.size()
        assert(D == 3)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        net_transformed = out3

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))
        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 1024)
        #
        if return_global:
            return out_max
        else:
            expand = out_max.view(-1, 1024, 1).repeat(1, 1, N)
            concat = torch.cat([point_cloud, expand, out1, out2, out3, out4, out5], 1)
            return concat, out_max


class PointFlowEncoder(nn.Module):
    def __init__(self, zdim, input_dim=3, use_deterministic_encoder=False):
        super(PointFlowEncoder, self).__init__()
        self.use_deterministic_encoder = use_deterministic_encoder
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)

        if self.use_deterministic_encoder:

            self.fc1 = nn.Linear(512, 512)
            self.fc2 = nn.Linear(512, 512)
            # self.fc_bn1 = nn.BatchNorm1d(512)
            # self.fc_bn2 = nn.BatchNorm1d(512)
            self.fc3 = nn.Linear(512, zdim)
        else:
            # Mapping to [c], cmean
            self.fc1_m = nn.Linear(512, 512)
            self.fc2_m = nn.Linear(512, 512)
            self.fc3_m = nn.Linear(512, zdim)
            # self.fc_bn1_m = nn.BatchNorm1d(512)
            # self.fc_bn2_m = nn.BatchNorm1d(v)

            # Mapping to [c], cmean
            self.fc1_v = nn.Linear(512, 512)
            self.fc2_v = nn.Linear(512, 512)
            self.fc3_v = nn.Linear(512, zdim)
            self.fc_bn1_v = nn.BatchNorm1d(512)
            self.fc_bn2_v = nn.BatchNorm1d(512)

    def forward(self, x):

        x = x - torch.mean(x, dim=1, keepdim=True)
        
        x = x.transpose(1, 2) ### bsz x 3 x n_pts
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        x = (self.conv4(x))

        x_glb = torch.max(x, 2, keepdim=True)[0] ## keepdim = True ## bsz 
        x_glb = x_glb.view(-1, 512)

        if self.use_deterministic_encoder:
            ms = F.relu((self.fc1(x_glb)))
            ms = F.relu((self.fc2(ms)))
            ms = self.fc3(ms)
            m, v = ms, 0
        else: ### bn for 
            m = F.relu((self.fc1_m(x_glb)))
            m = F.relu((self.fc2_m(m)))
            m = self.fc3_m(m)
            v = F.relu((self.fc1_v(x)))
            v = F.relu((self.fc2_v(v)))
            v = self.fc3_v(v)
        return x, x_glb 
