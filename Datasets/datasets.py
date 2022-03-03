import torch
import os
import numpy as np
import random
import pdb
import copy
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data.dataset import Dataset


class DatasetProcessing_MirFlickr25k(Dataset):
    def __init__(self, data_path, label_path, NN_tag, transformations, randnum, use='train'):
        self.img_path = data_path
        self.label_path = label_path
        self.NN_tag = NN_tag
        self.transforms = transformations

        with open(os.path.join(self.img_path, 'image_list.txt'), 'r') as f:
            img_list = [line.split()[0] for line in f]
        key_img = lambda s: int(s.split('.jpg')[0].split('im')[-1])
        fs_img = sorted(img_list, key=key_img)

        fs_label = [s for s in os.listdir(label_path) if "doc" not in s and "meta" not in s and "other" not in s]
        key_label = lambda s: s.split('.txt')[0]
        fs_label = sorted(fs_label, key=key_label)
        self.fs_label = [key_label(item) for item in fs_label]
        N_DATA = len([f for f in os.listdir(self.img_path) if '.jpg' in f])
        N_CLASS = len(fs_label)
        all_lab = np.load(os.path.join(self.label_path, "other", "labels_{}.npy".format(N_CLASS)))
        all_lab = list(all_lab)

        data = np.load(os.path.join(self.label_path, "other", "MIR_1kCT_expand.npz"), allow_pickle=True)
        self.W_dict = data['arr_0'].tolist()
        self.word2vec = data['arr_1']
        self.MIR_CT_NNw = data['arr_2'].tolist()
        self.MIR_CT_NNw2v = data['arr_3'].tolist()
        clean_imglist = data['arr_4'].tolist()
        tag_ind = data['arr_6'].tolist()

        # use randnum to divide dataset
        random.seed(randnum)
        random.shuffle(fs_img)
        random.seed(randnum)
        random.shuffle(all_lab)
        random.seed(randnum)
        random.shuffle(tag_ind)
        if use == 'train':
            self.img = fs_img[0:16000]
            self.all_lab = all_lab[0:16000]
            self.tag_ind = tag_ind[0:16000]
            self.rank_masks = torch.zeros(16000, self.word2vec.shape[0])
        else:
            self.img = fs_img[16000:18000]
            self.all_lab = all_lab[16000:18000]
            self.tag_ind = tag_ind[16000:18000]


    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img[index]))
        img = img.convert('RGB')
        label = torch.LongTensor(self.all_lab[index])

        tag_ind = copy.deepcopy(self.tag_ind[index])

        tag_vecs = self.word2vec[tag_ind, :]
        tag_ind = [1 if i in tag_ind else 0 for i in range(self.word2vec.shape[0])]

        tag_fusion = np.mean(tag_vecs, axis=0)

        tag_fusion = torch.tensor(tag_fusion, dtype=torch.float32)
        tag_ind = torch.Tensor(tag_ind)

        if isinstance(self.transforms, list):
            img1 = self.transforms[0](img)
            img2 = self.transforms[1](img)
            return img1, img2, label, tag_fusion, tag_ind, index
        else:
            img = self.transforms(img)
            return img, label, tag_fusion, tag_ind, index


    def img_show(self, img):
        _, fig = plt.subplots(1, 1)
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std*img + mean
        img = np.clip(img,0,1)
        fig.imshow(img)
        plt.show()

    
    def __len__(self):
        return len(self.img)






class DatasetProcessing_NUSwide(Dataset):
    def __init__(self, root, NN_tag, transformations, randnum, use='train'):
        self.root = root
        self.tag_path = os.path.join(root, 'NUS_WID_Tags', 'All_Tags.txt')
        self.NN_tag = NN_tag
        self.transforms = transformations

        data = np.load(os.path.join(root, 'WDHT/Nus21c_5018t_expand.npz'), allow_pickle=True)
        self.W_dict = data['arr_0'].tolist()
        self.word2vec = data['arr_1']
        self.CT_NNw = data['arr_2'].tolist()
        self.CT_NNw2v = data['arr_3'].tolist()

        with open(os.path.join(root, 'database_img.txt'), 'r') as f:
            img_list_db = [line.split()[0] for line in f]
        with open(os.path.join(root, 'test_img.txt'), 'r') as f:
            img_list_ts = [line.split()[0] for line in f]
        self.img_list = img_list_db + img_list_ts  
        clean_imglist = data['arr_4'].tolist()
        clean_imgindex = data['arr_5'].tolist()

        lab_fun = lambda s: list(map(int, s))
        with open(os.path.join(root, 'database_label_onehot.txt'), 'r') as fl:
            img_label_db = [lab_fun(line.split()) for line in fl]
        with open(os.path.join(root, 'test_label_onehot.txt'), 'r') as fl:
            img_label_ts = [lab_fun(line.split()) for line in fl] 
        self.img_label = img_label_db + img_label_ts
        clean_label = np.array(self.img_label)[clean_imgindex, :]
        clean_label = list(clean_label)
        
        tag_ind = np.array(data['arr_8']).tolist()
        clean_tag_ind = [tag_ind[i] for i in clean_imgindex]

        random.seed(randnum)
        random.shuffle(clean_imglist)
        random.seed(randnum)
        random.shuffle(clean_label)
        random.seed(randnum)
        random.shuffle(clean_tag_ind)

        if use == 'train':
            self.img = clean_imglist[0:100000]
            self.all_lab = clean_label[0:100000]
            self.tag_ind = clean_tag_ind[0:100000]
        else:
            self.img = clean_imglist[100000:102000]
            self.all_lab = clean_label[100000:102000]
            self.tag_ind = clean_tag_ind[100000:102000]
        
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.img[index]))
        img = img.convert('RGB')
        
        label = torch.LongTensor(self.all_lab[index])

        tag_ind = copy.deepcopy(self.tag_ind[index])
        
        tag_vecs = self.word2vec[tag_ind, :]
        tag_ind = [1 if i in tag_ind else 0 for i in range(self.word2vec.shape[0])]

        tag_fusion = np.mean(tag_vecs, axis=0)
        tag_fusion = torch.tensor(tag_fusion, dtype=torch.float32)
        tag_ind = torch.Tensor(tag_ind)
        
        if isinstance(self.transforms, list):
            img1 = self.transforms[0](img)
            img2 = self.transforms[1](img)
            return img1, img2, label, tag_fusion, tag_ind, index
        else:
            img = self.transforms(img)
            return img, label, tag_fusion, tag_ind, index
    

    def img_show(self, img):
        _, fig = plt.subplots(1, 1)
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std*img + mean
        img = np.clip(img,0,1)  # 以0~1为上下界进行截断
        fig.imshow(img)
        plt.show()


    def __len__(self):
        return len(self.img)
