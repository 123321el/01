#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       : dataloader and process data
@Author             : Kevinpro
@version            : 1.0
'''
import json
from transformers import BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

setup_seed(44)
def prepare_data():
    print("---Regenerate Data---")
    with open("train_data.json", 'r', encoding='utf-8') as load_f:
        info=[]
        import random
        for line in load_f.readlines():
            dic = json.loads(line)
            for j in dic['spo_list']:
                single_data={}
                single_data['rel']=j["predicate"]
                single_data['ent1']=j["object"]
                single_data['ent2'] = j["subject"]
                single_data['text']=dic['text']
                info.append(single_data)
        sub_train = info
    with open("train.json", "w",encoding='utf-8') as dump_f:
        for i in sub_train:
            a = json.dumps(i, ensure_ascii=False)
            dump_f.write(a)
            dump_f.write("\n")
    
    with open("dev_data.json", 'r', encoding='utf-8') as load_f:
        info=[]
        import random
        for line in load_f.readlines():
            dic = json.loads(line)
            for j in dic['spo_list']:
                single_data={}
                single_data['rel']=j["predicate"]
                single_data['ent1']=j["object"]
                single_data['ent2'] = j["subject"]
                single_data['text']=dic['text']
                info.append(single_data)
            
        sub_train = info
    with open("dev.json", "w",encoding='utf-8') as dump_f:
        for i in sub_train:
            a = json.dumps(i, ensure_ascii=False)
            dump_f.write(a)
            dump_f.write("\n")



def map_id_rel():    #加载关系标签
    id2rel={0: 'UNK', 1: '空间关系', 2: '部分关系', 3: '参数关系', 4: '材料优点', 5: '方案优点'}
    rel2id={}
    for i in id2rel:
        rel2id[id2rel[i]]=i
    return rel2id,id2rel

def load_train():
    rel2id,id2rel=map_id_rel()
    # max_length=512
    max_length = 256
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_data = {}
    train_data['label'] = []
    train_data['mask'] = []
    train_data['text'] = []

    with open("train.json", 'r', encoding='utf-8') as load_f:
        temp = load_f.readlines()
        # temp = temp[:5]                                     #可以选择训练多少条数据，注释后为全量数据
        for line in temp:
            dic = json.loads(line)
            if dic['rel'] not in rel2id:
                train_data['label'].append(0)
            else:
                train_data['label'].append(rel2id[dic['rel']])
            sent=dic['ent1']+dic['ent2']+dic['text']
            indexed_tokens = tokenizer.encode(sent, add_special_tokens=True)
            avai_len = len(indexed_tokens)
            while len(indexed_tokens) <  max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[: max_length]
            indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

            # Attention mask
            att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
            att_mask[0, :avai_len] = 1
            train_data['text'].append(indexed_tokens)
            train_data['mask'].append(att_mask)
    return train_data

def load_dev():
    rel2id,id2rel=map_id_rel()
    # max_length=512
    max_length = 256
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_data = {}
    train_data['label'] = []
    train_data['mask'] = []
    train_data['text'] = []

    with open("dev.json", 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            dic = json.loads(line)
            if dic['rel'] not in rel2id:
                train_data['label'].append(0)
            else:
                train_data['label'].append(rel2id[dic['rel']])

            sent=dic['ent1']+dic['ent2']+dic['text']
            indexed_tokens = tokenizer.encode(sent, add_special_tokens=True)


            

            avai_len = len(indexed_tokens)
            while len(indexed_tokens) <  max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[: max_length]
            indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

            # Attention mask
            att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
            att_mask[0, :avai_len] = 1
            train_data['text'].append(indexed_tokens)
            train_data['mask'].append(att_mask)
    return train_data

# if __name__ == '__main__':            #做数据预处理时使用
#     prepare_data()