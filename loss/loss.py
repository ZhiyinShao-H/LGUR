# -*- coding: utf-8 -*-
"""
Created on Sat., Aug. 3(rd), 2019 at 16:17

@author: zifyloo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init


class Loss(nn.Module):

    def __init__(self, opt):
        super(Loss, self).__init__()

        self.esp = 1e-8
        self.opt = opt

        self.W_Init()

    def W_Init(self):
        self.W = Parameter(torch.randn(512, self.opt.class_num))
        init.normal_(self.W.data, std=0.001)

    def calculate_CMPMLoss(self, image_embedding, text_embedding, label):

        Image_embedding_norm = image_embedding / image_embedding.norm(dim=1, keepdim=True)
        Text_embedding_norm = text_embedding / text_embedding.norm(dim=1, keepdim=True)

        image_text = torch.mm(image_embedding, Text_embedding_norm.t())
        text_image = torch.mm(text_embedding, Image_embedding_norm.t())
        P_image2text = F.softmax(image_text, dim=1)
        P_text2image = F.softmax(text_image, dim=1)

        labels_reshape = torch.reshape(label, (label.size(0), 1))
        labels_dist = labels_reshape - labels_reshape.t()
        y = (labels_dist == 0)
        Q = y.float() / y.float().sum(dim=1, keepdim=True)

        '''torch.log(P_image2text / (Q + self.esp)) will cause training collapse'''
        Li2t = torch.mean(torch.sum(P_image2text *
                                    (F.log_softmax(image_text, dim=1) - torch.log(Q + self.esp)), dim=1))
        Lt2i = torch.mean(torch.sum(P_text2image *
                                    (F.log_softmax(text_image, dim=1) - torch.log(Q + self.esp)), dim=1))

        CMPM_loss = Li2t + Lt2i

        sim_cos = torch.matmul(Image_embedding_norm, Text_embedding_norm.t())

        positive_sim = torch.mean(sim_cos[y])
        negative_sim = torch.mean(sim_cos[y == 0])

        return CMPM_loss, positive_sim, negative_sim

    def calculate_CMPCLoss(self, image_embedding, text_embedding, label):
        criterion = nn.CrossEntropyLoss(reduction='mean')
        self.W_norm = self.W / self.W.norm(dim=0, keepdim=True)

        Image_embedding_norm = image_embedding / image_embedding.norm(dim=1, keepdim=True)
        Text_embedding_norm = text_embedding / text_embedding.norm(dim=1, keepdim=True)

        z_cap_i2t = torch.sum(image_embedding * Text_embedding_norm, dim=1, keepdim=True) * Text_embedding_norm
        z_cap_t2i = torch.sum(text_embedding * Image_embedding_norm, dim=1, keepdim=True) * Image_embedding_norm

        score_i2t = torch.mm(z_cap_i2t, self.W_norm)
        score_t2i = torch.mm(z_cap_t2i, self.W_norm)

        label = label.view(label.size(0))
        Lipt = criterion(score_i2t, label)
        Ltpi = criterion(score_t2i, label)

        CMPCLoss = Lipt + Ltpi

        pred_i2t = torch.mean((torch.argmax(score_i2t, dim=1) == label).float())
        pred_t2i = torch.mean((torch.argmax(score_t2i, dim=1) == label).float())

        return CMPCLoss, pred_i2t, pred_t2i

    def forward(self, image_embedding, text_embedding, label):

        CMPC_loss = 0.0
        CMPM_loss = 0.0
        pred_i2t = 0.0
        pred_t2i = 0.0
        negative_sim = 0.0
        positive_sim = 0.0

        if self.opt.CMPM:
            CMPM_loss, positive_sim, negative_sim = self.calculate_CMPMLoss(image_embedding, text_embedding, label)
        if self.opt.CMPC:
            CMPC_loss, pred_i2t, pred_t2i = self.calculate_CMPCLoss(image_embedding, text_embedding, label)

        # loss = CMPM_loss + CMPC_loss

        return CMPM_loss, CMPC_loss, pred_i2t, pred_t2i, positive_sim, negative_sim


"""
class Loss_master(nn.Module):
    def __init__(self, args):
        super(Loss_master, self).__init__()
        self.CMPM = args.CMPM
        self.CMPC = args.CMPC
        self.epsilon = 1e-8
        self.num_classes = args.class_num

        self.W = Parameter(torch.randn(512, args.class_num))
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.W.data, gain=1)

    def compute_cmpc_loss(self, image_embeddings, text_embeddings, labels):
        
        criterion = nn.CrossEntropyLoss(reduction='mean')
        self.W_norm = self.W / self.W.norm(dim=0)
        # labels_onehot = one_hot_coding(labels, self.num_classes).float()
        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        image_proj_text = torch.sum(image_embeddings * text_norm, dim=1, keepdim=True) * text_norm
        text_proj_image = torch.sum(text_embeddings * image_norm, dim=1, keepdim=True) * image_norm

        image_logits = torch.matmul(image_proj_text, self.W_norm)
        text_logits = torch.matmul(text_proj_image, self.W_norm)

        # labels_one_hot = one_hot_coding(labels, num_classes)
        '''
        ipt_loss = criterion(input=image_logits, target=labels)
        tpi_loss = criterion(input=text_logits, target=labels)
        cmpc_loss = ipt_loss + tpi_loss
        '''
        labels = labels.view(labels.size(0))
        cmpc_loss = criterion(image_logits, labels) + criterion(text_logits, labels)
        # cmpc_loss = - (F.log_softmax(image_logits, dim=1) + F.log_softmax(text_logits, dim=1)) * labels_onehot
        # cmpc_loss = torch.mean(torch.sum(cmpc_loss, dim=1))
        # classification accuracy for observation
        image_pred = torch.argmax(image_logits, dim=1)
        text_pred = torch.argmax(text_logits, dim=1)

        image_precision = torch.mean((image_pred == labels).float())
        text_precision = torch.mean((text_pred == labels).float())

        return cmpc_loss, image_precision, text_precision

    def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels):
       
        batch_size = image_embeddings.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)

        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        image_proj_text = torch.matmul(image_embeddings, text_norm.t())
        text_proj_image = torch.matmul(text_embeddings, image_norm.t())

        # normalize the true matching distribution
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

        i2t_pred = F.softmax(image_proj_text, dim=1)
        # i2t_loss = i2t_pred * torch.log((i2t_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + self.epsilon))

        t2i_pred = F.softmax(text_proj_image, dim=1)
        # t2i_loss = t2i_pred * torch.log((t2i_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + self.epsilon))

        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        sim_cos = torch.matmul(image_norm, text_norm.t())

        pos_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask))
        neg_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask == 0))

        return cmpm_loss, pos_avg_sim, neg_avg_sim

    def calculate_CMPMLoss(self, image_embedding, text_embedding, label):

        Image_embedding_norm = image_embedding / image_embedding.norm(dim=1, keepdim=True)
        Text_embedding_norm = text_embedding / text_embedding.norm(dim=1, keepdim=True)

        image_text = torch.mm(image_embedding, Text_embedding_norm.t())
        text_image = torch.mm(text_embedding, Image_embedding_norm.t())
        P_image2text = F.softmax(image_text, dim=1)
        P_text2image = F.softmax(text_image, dim=1)

        labels_reshape = torch.reshape(label, (label.size(0), 1))
        labels_dist = labels_reshape - labels_reshape.t()
        y = (labels_dist == 0)
        Q = y.float() / y.float().sum(dim=1, keepdim=True)

        '''torch.log(P_image2text / (Q + self.esp)) will cause training collapse'''
        Li2t = torch.mean(torch.sum(P_image2text * (F.log_softmax(image_text, dim=1) - torch.log(Q + self.epsilon)), dim=1))
        Lt2i = torch.mean(torch.sum(P_text2image * (F.log_softmax(text_image, dim=1) - torch.log(Q + self.epsilon)), dim=1))

        CMPM_loss = Li2t + Lt2i

        sim_cos = torch.matmul(Image_embedding_norm, Text_embedding_norm.t())

        positive_sim = torch.mean(sim_cos[y])
        negative_sim = torch.mean(sim_cos[y == 0])

        return CMPM_loss, positive_sim, negative_sim

    def calculate_CMPCLoss(self, image_embedding, text_embedding, label):
        criterion = nn.CrossEntropyLoss(reduction='mean')
        self.W_norm = self.W / self.W.norm(dim=0, keepdim=True)

        Image_embedding_norm = image_embedding / image_embedding.norm(dim=1, keepdim=True)
        Text_embedding_norm = text_embedding / text_embedding.norm(dim=1, keepdim=True)

        z_cap_i2t = torch.sum(image_embedding * Text_embedding_norm, dim=1, keepdim=True) * Text_embedding_norm
        z_cap_t2i = torch.sum(text_embedding * Image_embedding_norm, dim=1, keepdim=True) * Image_embedding_norm

        score_i2t = torch.mm(z_cap_i2t, self.W_norm)
        score_t2i = torch.mm(z_cap_t2i, self.W_norm)

        label = label.view(label.size(0))
        Lipt = criterion(score_i2t, label)
        Ltpi = criterion(score_t2i, label)

        CMPCLoss = Lipt + Ltpi

        pred_i2t = torch.mean((torch.argmax(score_i2t, dim=1) == label).float())
        pred_t2i = torch.mean((torch.argmax(score_t2i, dim=1) == label).float())

        return CMPCLoss, pred_i2t, pred_t2i

    def forward(self, image_embeddings, text_embeddings, labels):
        cmpm_loss = 0.0
        cmpc_loss = 0.0
        image_precision = 0.0
        text_precision = 0.0
        neg_avg_sim = 0.0
        pos_avg_sim = 0.0
        if self.CMPM:
            print(self.compute_cmpm_loss(image_embeddings, text_embeddings, labels))
            print(self.calculate_CMPMLoss(image_embeddings, text_embeddings, labels))
            cmpm_loss, pos_avg_sim, neg_avg_sim = self.compute_cmpm_loss(image_embeddings, text_embeddings, labels)
        if self.CMPC:
            print(self.compute_cmpc_loss(image_embeddings, text_embeddings, labels))
            print(self.calculate_CMPCLoss(image_embeddings, text_embeddings, labels))
            cmpc_loss, image_precision, text_precision = self.compute_cmpc_loss(image_embeddings, text_embeddings,
                                                                                labels)

        return cmpm_loss, cmpc_loss, image_precision, text_precision, pos_avg_sim, neg_avg_sim
"""
"""
from option.options import options
import numpy
import tqdm

for i in tqdm.tqdm(range(5000)):
    numpy.random.seed(i)
    torch.manual_seed(i)

    labels = torch.randint(11003, [16])
    image_feature = torch.rand((16, 512))
    text_feature = torch.rand((16, 512))

    opt = options().opt

    loss2 = Loss2(opt)
    loss1 = Loss(opt)
    cmpm_loss2, _, _, _, _, _ = loss2(image_feature, text_feature, labels)
    cmpm_loss1, _, _, _, _, _ = loss1(image_feature, text_feature, labels)
    if cmpm_loss1 - cmpm_loss2 != 0:
        print(cmpm_loss2, cmpm_loss1)

"""

"""
from option.options import options
import numpy
import tqdm

numpy.random.seed(233)
torch.manual_seed(233)

labels = torch.randint(11003, [16])
image_feature = torch.rand((16, 512))
text_feature = torch.rand((16, 512))

opt = options().opt
loss = Loss(opt)
loss(image_feature, text_feature, labels)

import numpy as np
x = [i for i in range(100, 200, 10)]
print(x)
"""