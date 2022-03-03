import os
import numpy as np
import gensim
import pdb

from tqdm import tqdm

root = '.../mirflickr25k_annotations_v080'
data_path = '.../mirflickr'


CT_file = 'other/common_tags.txt'
word2vec_model_path = 'word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin'
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> loading word2vec model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
with open(os.path.join(root, CT_file), 'r') as f:
    common_tags = [line.split()[0] for line in f]
print("common tags:{}, common tag num:{}".format(common_tags, len(common_tags)))

W_MIR = dict()
MIR_CT_w2v = []
MIR_CT_NNw = []
MIR_CT_NNw2v = []
for i, word in enumerate(tqdm(common_tags)):
    W_MIR[word] = i
    nn_words, nn_w2v = [], []
    try:
        MIR_CT_w2v.append(word2vec_model[word])
        items = word2vec_model.most_similar(word)
        for item in items:
            nn_words.append(item[0])
            nn_w2v.append(word2vec_model[item[0]])
        nn_w2v = np.array(nn_w2v)
        assert(len(nn_words) == 10)
    except:
        MIR_CT_w2v.append(np.random.uniform(-0.25, 0.25, 300))
        nn_words = None
        nn_w2v = None
    MIR_CT_NNw.append(nn_words)
    MIR_CT_NNw2v.append(nn_w2v)

MIR_CT_w2v = np.array(MIR_CT_w2v)
print("nus_5kCT_w2v:{}".format(MIR_CT_w2v.shape))
print("W_MIR:{}, MIR_CT_NNw:{}".format(list(W_MIR)[456], MIR_CT_NNw[456]))
assert(MIR_CT_w2v.shape == (1386, 300))


tag_path = os.path.join(data_path, 'meta', 'tags')
with open(os.path.join(tag_path, 'tag_list.txt'), 'r') as f:
    fs_tags = [line.split()[0] for line in f]
fs_tags = [os.path.join(tag_path, f) for f in fs_tags]
print("fs_tags:{}".format(len(fs_tags)))

clean_imglist = []
noCT_imglist = []
MIR_CT_inds = []
for i, tags in enumerate(tqdm(fs_tags)):
    with open(tags, 'r') as f:
        current_tags = [line.strip() for line in f]
    CT_index = []
    for j, word in enumerate(current_tags):
        try:
            p = W_MIR[word]
            CT_index.append(p)
        except:
            pass
    if len(CT_index) != 0:
        clean_imglist.append(tags)
    else:
        noCT_imglist.append(tags)
    MIR_CT_inds.append(CT_index)
print("no tag images:{}".format(len(noCT_imglist)))
print("MIR_CT_bvec length:{}".format(len(MIR_CT_inds)))

np.savez(os.path.join(root, 'other/MIR_1kCT_expand.npz'), common_tags, MIR_CT_w2v, MIR_CT_NNw, MIR_CT_NNw2v, clean_imglist, noCT_imglist, MIR_CT_inds)