
import os
import numpy as np
import gensim
import pdb

from tqdm import tqdm

data_path = '.../nus-wide'
root = '.../nus-wide/other'
taglist5k_file = 'tags_nus.npy'
tagdict_file = 'W_nus_dict.npy'


word2vec_model_path = 'word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin'
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> loading word2vec model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
tags_nus = np.load(os.path.join(root, taglist5k_file), allow_pickle=True)
tags_nus = list(tags_nus)

Nus_5kCT_w2v = []
Nus_5kCT_NNw = []
Nus_5kCT_NNw2v = []
for i, word in enumerate(tqdm(tags_nus)):
    nn_words, nn_w2v = [], []
    try:
        Nus_5kCT_w2v.append(word2vec_model[word])
        items = word2vec_model.most_similar(word)
        for item in items:
            nn_words.append(item[0])
            nn_w2v.append(word2vec_model[item[0]])
        nn_w2v = np.array(nn_w2v)
        assert(len(nn_words) == 10)
        assert(nn_w2v.shape == (10, 300))
    except:
        Nus_5kCT_w2v.append(np.random.uniform(-0.25, 0.25, 300))
        nn_words = None
        nn_w2v = None
    Nus_5kCT_NNw.append(nn_words)
    Nus_5kCT_NNw2v.append(nn_w2v)

Nus_5kCT_w2v = np.array(Nus_5kCT_w2v)
print("nus_5kCT_w2v:{}".format(Nus_5kCT_w2v.shape))
print("Nus_tag:{}, Nus_NNw:{}".format(tags_nus[236], Nus_5kCT_NNw[236]))
assert(Nus_5kCT_w2v.shape == (5018, 300))



count = []
all_I2T_dict = dict()
with open(os.path.join(data_path, 'NUS_WID_Tags', 'All_Tags.txt'), 'r') as ft:
    for line in ft:
        tmp = line.split()
        all_I2T_dict[tmp[0]] = tmp[1:]
        count.append(tmp[0])
print(len(all_I2T_dict))


with open(os.path.join(data_path, 'database_img.txt'), 'r') as f:
    img_list_db = [line.split()[0] for line in f]  
with open(os.path.join(data_path, 'test_img.txt'), 'r') as f:
    img_list_ts = [line.split()[0] for line in f]  
img_list = img_list_db + img_list_ts 
print(">>>>>>>>>>images list length:{}, images:{}>>>>>>>>>>>".format(len(img_list), img_list[0:10]))
assert(len(img_list) == 195834)


lab_fun = lambda s: list(map(int, s))
with open(os.path.join(data_path, 'database_label_onehot.txt'), 'r') as fl:
    img_label_db = [lab_fun(line.split()) for line in fl]  
with open(os.path.join(data_path, 'test_label_onehot.txt'), 'r') as fl:
    img_label_ts = [lab_fun(line.split()) for line in fl]
img_label = img_label_db + img_label_ts
print(">>>>>>>>>>label list length:{}, label:{}>>>>>>>>>>>".format(len(img_label), img_label[0:10]))
img_label = np.array(img_label)
assert(img_label.shape == (195834, 21))
label_sum = np.sum(img_label, axis=1)
for i in range(len(label_sum)):
    if label_sum[i] == 0:
        print("no label image:", img_list[i], img_label[i])



Nus21_5kCT_inds = []
clean_imglist = []
clean_imgindex = []
noCT_imglist = []
noCT_imgindex = []
W_nus_dict = np.load(os.path.join(root, tagdict_file), allow_pickle=True).item()
key = lambda s: s.split('_')[1].split('.')[0]
for i, img in enumerate(tqdm(img_list)):
    current_tags = all_I2T_dict[key(img)]
    CT_index = []
    for j, word in enumerate(current_tags):
        try:
            p = W_nus_dict[word]
            CT_index.append(p)
        except:
            pass
    if len(CT_index) != 0:
        clean_imglist.append(img)
        clean_imgindex.append(i)
    else:
        noCT_imglist.append(img)
        noCT_imgindex.append(i)
    Nus21_5kCT_inds.append(CT_index)
print("no tag images:{}".format(len(noCT_imglist)))
print("nus21_5kCT_bvec length:{}".format(len(Nus21_5kCT_inds)))
assert(len(Nus21_5kCT_inds) == 195834)

np.savez(os.path.join(root, 'Nus21c_5018t_expand.npz'), tags_nus, Nus_5kCT_w2v, Nus_5kCT_NNw, Nus_5kCT_NNw2v, clean_imglist, clean_imgindex, noCT_imglist, noCT_imgindex, Nus21_5kCT_inds)