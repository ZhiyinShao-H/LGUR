from option.options import options, config
from data.dataloader import get_dataloader
import torch
import random
from model.model import TextImgPersonReidNet
from loss.Id_loss import Id_Loss
from loss.RankingLoss import RankingLoss
from torch import optim
import logging
import os
from test_during_train import test , test_part
from torch.autograd import Variable
from model.DETR_model import TextImgPersonReidNet_mydecoder_pixelVit_transTXT_3, TextImgPersonReidNet_mydecoder_pixelVit_transTXT_3_vit
import torch.nn as nn
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def save_checkpoint(state, opt):

    filename = os.path.join(opt.save_path, 'model/best.pth.tar')
    torch.save(state, filename)


def load_checkpoint(opt):
    filename = os.path.join(opt.save_path, 'model/best.pth.tar')
    state = torch.load(filename)

    return state


def calculate_similarity(image_embedding, text_embedding):
    image_embedding_norm = image_embedding / image_embedding.norm(dim=1, keepdim=True)
    text_embedding_norm = text_embedding / text_embedding.norm(dim=1, keepdim=True)

    similarity = torch.mm(image_embedding_norm, text_embedding_norm.t())

    return similarity

def calculate_similarity_part(numpart,image_embedding, text_embedding):
    image_embedding = torch.cat([image_embedding[i] for i in range(numpart)],dim=1)
    text_embedding = torch.cat([text_embedding[i] for i in range(numpart)], dim=1)
    image_embedding_norm = image_embedding / image_embedding.norm(dim=1, keepdim=True)
    text_embedding_norm = text_embedding / text_embedding.norm(dim=1, keepdim=True)

    similarity = torch.mm(image_embedding_norm, text_embedding_norm.t())

    return similarity

def calculate_similarity_score(opt,image_embedding, text_embedding , img_score ,txt_score):
    img_size = img_score.size(0)
    txt_size = txt_score.size(0)
    part_num = img_score.size(1)
    Final_matrix = torch.FloatTensor(img_size, txt_size).zero_().to(opt.device)
    Fq_matrix = torch.FloatTensor(img_size, txt_size).zero_().to(opt.device)
    for i in range(part_num):
        # print(i)
        # Compute pairwise distance, replace by the official when merged
        image_embedding_i = image_embedding[i]
        text_embedding_i = text_embedding[i]
        image_embedding_i = image_embedding_i / image_embedding_i.norm(dim=1, keepdim=True)
        text_embedding_i = text_embedding_i / text_embedding_i.norm(dim=1, keepdim=True)
        similarity = torch.mm(image_embedding_i, text_embedding_i.t())
        img_score_i = img_score[:, i].unsqueeze(1)  # .view(q_score.size(0), 1)
        txt_score_i = txt_score[:, i].unsqueeze(1)
        # print(img_score.shape)
        # print(img_score_i.shape)
        q_matrix = torch.mm(img_score_i, txt_score_i.t())
        final_matrix = similarity.mul(q_matrix)
        Final_matrix = Final_matrix + final_matrix
        Fq_matrix = Fq_matrix + q_matrix
    Fq_matrix = Fq_matrix + 1e-12
    # print(Fq_matrix)
    dist_part = torch.div(Final_matrix, Fq_matrix)

    return dist_part

if __name__ == '__main__':
    opt = options().opt
    opt.GPU_id = '1'
    opt.device = torch.device('cuda:{}'.format(opt.GPU_id))
    opt.data_augment = False
    opt.lr = 0.001
    opt.margin = 0.2

    opt.feature_length = 512

    opt.train_dataset = 'CUHK-PEDES'
    opt.dataset = 'MSMT-PEDES'

    if opt.dataset == 'MSMT-PEDES':
        opt.pkl_root = '/data1/zhiying/text-image/MSMT-PEDES/3-1/'
        opt.class_num = 3102
        opt.vocab_size = 2500
        opt.dataroot = '/data1/zhiying/text-image/data/ICFG_PEDES'
        # opt.class_num = 2802
        # opt.vocab_size = 2300
    elif opt.dataset == 'CUHK-PEDES':
        opt.pkl_root = '/data1/zhiying/text-image/CUHK-PEDES_/'  # same_id_new_
        opt.class_num = 11003
        opt.vocab_size = 5000
        opt.dataroot = '/data1/zhiying/text-image/CUHK-PEDES'

    opt.d_model = 1024
    opt.nhead = 4
    opt.dim_feedforward = 2048
    opt.normalize_before = False
    opt.num_encoder_layers = 3
    opt.num_decoder_layers = 3
    opt.num_query = 6
    opt.detr_lr = 0.0001
    opt.txt_detr_lr = 0.0001
    opt.txt_lstm_lr = 0.001
    opt.res_y = False
    opt.noself = False
    opt.post_norm = False
    opt.n_heads = 4
    opt.n_layers = 2
    opt.share_query = True
    model_name = 'random_my_small_DeiT_2version_head6_384lstm'
    # model_name = 'test'
    opt.save_path = './checkpoints/dual_modal/{}/'.format(opt.train_dataset) + model_name

    opt.epoch = 60
    opt.epoch_decay = [20, 40, 50]

    opt.batch_size = 64
    opt.start_epoch = 0
    opt.trained = False

    config(opt)
    opt.epoch_decay = [i - opt.start_epoch for i in opt.epoch_decay]

    train_dataloader = get_dataloader(opt)
    opt.mode = 'test'
    test_img_dataloader, test_txt_dataloader = get_dataloader(opt)
    opt.mode = 'train'
    # train_dataloader = get_dataloader(opt)
    # opt.mode = 'test'
    # test_img_dataloader, test_txt_dataloader = get_dataloader(opt)
    # opt.mode = 'train'

    id_loss_fun = nn.ModuleList()
    for _ in range(opt.num_query):
        id_loss_fun.append(Id_Loss(opt).to(opt.device))
    ranking_loss_fun = RankingLoss(opt)
    network = TextImgPersonReidNet_mydecoder_pixelVit_transTXT_3_vit(opt).to(opt.device)
    test_best = 0
    test_history = 0
    state = load_checkpoint(opt)
    network.load_state_dict(state['network'])
    test_best = state['test_best']
    test_history = test_best
    print('load the {} epoch param successfully'.format(state['epoch']))
    """
    network.eval()
    test_best = test(opt, 0, 0, network,
                     test_img_dataloader, test_txt_dataloader, test_best)
    network.train()
    exit(0)
    """
    network.eval()
    test_best = test_part(opt, state['epoch'], 1, network,
                          test_img_dataloader, test_txt_dataloader, test_best)
    logging.info('Training Done')





