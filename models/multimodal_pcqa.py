import torch
import torch.nn as nn
from models.backbones import pointnet2,resnet50
from models.transformer import TransformerEncoderLayer_CMA


class CMA_fusion(nn.Module):
    def __init__(self, img_inplanes, pc_inplanes, cma_planes = 1024):
        super(CMA_fusion, self).__init__()
        self.encoder = TransformerEncoderLayer_CMA(d_model = cma_planes, nhead = 8, dim_feedforward = 2048, dropout = 0.1)
        self.linear1 = nn.Linear(img_inplanes,cma_planes)
        self.linear2 = nn.Linear(pc_inplanes,cma_planes)
        self.quality1 = nn.Linear(cma_planes * 4, cma_planes * 2)
        self.quality2 = nn.Linear(cma_planes * 2,1)
        self.img_bn = nn.BatchNorm1d(cma_planes)
        self.pc_bn = nn.BatchNorm1d(cma_planes)         
   
    def forward(self, img, pc):
        # linear mapping and batch normalization
        img = self.linear1(img)
        img = self.img_bn(img)
        pc = self.linear2(pc)
        pc = self.pc_bn(pc)
        # cross modal attention and feature fusion
        img = img.unsqueeze(0)
        pc = pc.unsqueeze(0)
        img_a,pc_a = self.encoder(img,pc)
        output = torch.cat((img,img_a,pc_a,pc), dim=2)
        output = output.squeeze(0)
        # feature regression
        output = self.quality1(output)
        output = self.quality2(output)
        return output



class MM_PCQAnet(nn.Module):
    def __init__(self):
        super(MM_PCQAnet, self).__init__()
        self.img_inplanes = 2048
        self.pc_inplanes = 1024
        self.cma_planes = 1024
        self.img_backbone = resnet50(pretrained=True)
        self.pc_backbone = pointnet2()
        self.regression = CMA_fusion(img_inplanes = self.img_inplanes, pc_inplanes = self.pc_inplanes, cma_planes = self.cma_planes)          
   
    def forward(self, img, pc):
        # extract features from the projections
        img_size = img.shape
        img = img.view(-1, img_size[2], img_size[3], img_size[4])
        img = self.img_backbone(img)
        img = torch.flatten(img, 1)
        # average the projection features
        img = img.view(img_size[0],img_size[1],self.img_inplanes)
        img = torch.mean(img, dim = 1)
        
        # extract features from patches
        pc_size = pc.shape
        pc = pc.view(-1,pc_size[2],pc_size[3])
        pc = self.pc_backbone(pc)
        # average the patch features
        pc = pc.view(pc_size[0],pc_size[1],self.pc_inplanes)
        pc = torch.mean(pc, dim = 1)
        # attention, fusion, and regression
        output = self.regression(img,pc)

        return output


