
import os
import torch
import argparse
import numpy as np
import pickle
import copy

from torchvision import transforms, models
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm

from Datasets.datasets import DatasetProcessing_MirFlickr25k, DatasetProcessing_NUSwide
from Modules.models import retrieval_model, WSD_model
from utils.utils import GenerateCode, CalcMap, CalcMap_at_k 
from Modules.warmup_scheduler import GradualWarmupScheduler
from Modules.layers import Hinge_loss


def setup_seed(seed=8):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_optimizer(model, lr, lr_ft, lr_d, warm_up, type=None):
    if type == 'img':
        optimizer = torch.optim.SGD([
            {'params': model.feature.parameters()},
            {'params': model.classifier.parameters()},
            {'params': model.fc3.parameters(), 'lr': lr, 'momentum': 0.9},
            {'params': model.H1.parameters(), 'lr': lr, 'momentum': 0.9},
            ], lr = lr_ft, momentum = 0.9)
    elif type == 'tag':
        optimizer = torch.optim.SGD([
            {'params': model.TR_module.parameters()},
            ], lr=lr_d, momentum = 0.9)
    else:
        optimizer = None

    if warm_up:
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warm_up, after_scheduler=None)
    else:
        scheduler_warmup = None

    return optimizer, scheduler_warmup



def WSDRCA(args):
    test_transformations = transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_1 = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomCrop(227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_2 = transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_transformations = [transform_1, transform_2]


    original_model = models.alexnet(pretrained=True)
    bit = args.bits
    d_model = 300
    heads = 3
    d_ff = 512
    model = WSD_model(original_model, d_model, heads, d_ff, bit)
    model.cuda()

    num_workers = 8
    NN_tag = 0
    Del_num = 0
    batch_sz = 32
    lr_ft = 1e-4
    lr = 1e-3
    lr_d = 2e-4
    warm_up = 20
    margin = args.margin
    lam1, lam2, lam3 = 1, 10, 1

    upd_flag = [0,0,1,1]*20
    epochs = len(upd_flag)
    
    print(model)
    

    if args.dataset == 'MirFlickr':
        data_path = '.../mirflickr' 
        label_path = '.../mirflickr25k_annotations_v080'
        randnum = 66
        train_set = DatasetProcessing_MirFlickr25k(data_path, label_path, NN_tag, train_transformations, randnum, use='train')
        database_set = DatasetProcessing_MirFlickr25k(data_path, label_path, NN_tag, test_transformations, randnum, use='train')
        test_set = DatasetProcessing_MirFlickr25k(data_path, label_path, NN_tag, test_transformations, randnum, use='test')
        n_class = 38
    else:
        data_path = '.../nus-wide'
        randnum = 25
        train_set = DatasetProcessing_NUSwide(data_path, NN_tag, train_transformations, randnum, use='train')
        database_set = DatasetProcessing_NUSwide(data_path, NN_tag, test_transformations, randnum, use='train')
        test_set = DatasetProcessing_NUSwide(data_path, NN_tag, test_transformations, randnum, use='test')
        n_class = 21
    
    tag_optimizer, tag_scheduler = init_optimizer(model, lr, lr_ft, lr_d, warm_up, type='tag')
    img_optimizer, img_scheduler = init_optimizer(model, lr, lr_ft, lr_d, warm_up, type='img')

    num_train = len(train_set)
    train_loader = DataLoader(train_set,
                                batch_size=batch_sz,
                                shuffle=True,
                                num_workers=num_workers
                                )
    num_database = len(database_set)
    database_loader = DataLoader(database_set,
                                batch_size=batch_sz,
                                shuffle=True,
                                num_workers=num_workers
                                )                     
    num_test = len(test_set)
    test_loader = DataLoader(test_set,
                                batch_size=batch_sz,
                                shuffle=True,
                                num_workers=num_workers
                                )

    train_loss = []
    map_record = []
    mapk_record = []
    final_map = 0
    final_map_at_k = 0

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> experiment settings >>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("dataset:{}, bit:{}, epochs:{}, batch_sz:{}, NN_tag:{}, Del_num:{}, lr_ft:{}, lr:{}, lr_d:{}".format(
        args.dataset, bit, epochs, batch_sz, NN_tag, Del_num, lr_ft, lr, lr_d))
    print("warm-up:{}, margin:{}, lam1:{}, lam2:{}, lam3:{}, randnum:{}".format(
        warm_up, margin, lam1, lam2, lam3, randnum))
    print("tag refinement module d_model:{}, heads:{}, d_ff:{}".format(
        d_model, heads, d_ff))
    

    word2vec = torch.Tensor(train_set.word2vec).cuda()
    
    for epoch in range(epochs):
        # training stage
        epoch_loss = 0
        model.train()
        for i, train_data in enumerate(tqdm(train_loader)):
            img1, img2, _, _, tag_inds, index = train_data
            if upd_flag[epoch] == 1:
                imgs, tag_inds = img1.cuda(), tag_inds.cuda()  # 较多的数据增强
            else:
                imgs, tag_inds = img2.cuda(), tag_inds.cuda()
            N = len(imgs)
            h, f, psd_label, attn_w1, attn_w2 = model(imgs, word2vec, tag_inds)

            # L1: pair-wise loss
            difference = torch.unsqueeze(h, 0) - torch.unsqueeze(h, 1)
            ps_s = torch.div(psd_label, torch.norm(psd_label, p=2, dim=1).unsqueeze(1) + 1e-15)  # 计算余弦相似度
            ps_s = ps_s.matmul(ps_s.t())  # N × N
            L1 = (torch.abs((torch.sum(difference*difference, 2)/bit + (ps_s - 1)/2))).mean()

            # L2: hinge loss
            L2 = Hinge_loss(f, psd_label, margin)

            # L3: quantization loss
            L3 = - torch.sum((h - 0.5)*(h - 0.5)/bit) / N

            Loss = lam1*L1 + lam2*L2 + lam3*L3
            # print("L1:{}, L2:{}, L3:{}, Loss:{}".format(L1, L2, L3, Loss))

            if upd_flag[epoch] == 0:
                tag_optimizer.zero_grad()
                Loss.backward()
                tag_optimizer.step()
                curr_optimizer = tag_optimizer
                curr_scheduler = tag_scheduler
            elif upd_flag[epoch] == 1:
                img_optimizer.zero_grad()
                Loss.backward()
                img_optimizer.step()
                curr_optimizer = img_optimizer
                curr_scheduler = img_scheduler
            else:
                print("wrong flag")
            epoch_loss += Loss

        train_loss.append(epoch_loss / len(train_loader))
        print("[training phase][Epoch:%3d/%3d][Loss:%3.5f][lr_ft:%1.7f][lr:%1.7f]"%(
            epoch+1, args.epochs, epoch_loss/len(train_loader), curr_optimizer.state_dict()['param_groups'][0]['lr'], curr_optimizer.state_dict()['param_groups'][-1]['lr']))


        # testing stage
        if upd_flag[epoch] != 0:
            model.eval()
            test_model = retrieval_model(model)
            test_model.cuda()
            test_model.eval()
            dB, d_labels = GenerateCode(test_model, database_loader, num_database, bit, batch_sz, n_class)
            qB, q_labels = GenerateCode(test_model, test_loader, num_test, bit, batch_sz, n_class)
            if args.dataset == 'Nuswide':
                map = CalcMap_at_k(qB, dB, q_labels, d_labels, 50000)
            else:
                map = CalcMap(qB, dB, q_labels, d_labels)
            map_at_k = CalcMap_at_k(qB, dB, q_labels, d_labels, 5000)
            map_record.append(map)
            mapk_record.append(map_at_k)
            if final_map < map:
                final_map = map
                best_test_model1 = copy.deepcopy(test_model)
            if final_map_at_k < map_at_k:
                final_map_at_k = map_at_k
                best_test_model2 = copy.deepcopy(test_model)  
            print('[Test Phase][Epoch: %3d/%3d] current mAP, mAP@K: %3.5f, %3.5f; highest mAP, mAP@k: %3.5f, %3.5f' % (epoch+1, epochs, map, map_at_k, final_map, final_map_at_k))
            

        if curr_scheduler is not None:
            curr_scheduler.step()  # warm-up学习率衰减
    
    return final_map, final_map_at_k, best_test_model1, best_test_model2


if __name__=="__main__":
    parser = argparse.ArgumentParser("transformer attention")
    parser.add_argument('--dataset', type=str, default='MirFlickr', choices=['MirFlickr', 'Nuswide'], help="choose dataset")
    parser.add_argument('--margin', type=float, default=None, help="choose margin")
    parser.add_argument('--bits', type=int, default=None, choices=[16, 24, 32, 48], help="choose bits")
    parser.add_argument('--device', type=str, default=None, choices=['0', '1', '2', '3'], help="choose device")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    map, mpk, test_model1, test_model2 = WSDRCA(args)

    result = {}
    result['map'] = map
    result['mapk'] = mpk
    result['test_model_dict1'] = test_model1.state_dict()
    result['test_model_dict2'] = test_model2.state_dict()
    result['filename'] = 'results/WSDRCA' + str(args.bits) + 'bits_' + args.dataset + \
                         datetime.now().strftime("_%y-%m-%d-%H-%M-%S") + '.pkl'

    print('---------------------------------------')
    fp = open(result['filename'], 'wb')
    pickle.dump(result, fp)
    fp.close()


    


        
    




