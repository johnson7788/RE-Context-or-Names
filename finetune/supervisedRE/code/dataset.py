
import os 
import re
import ast 
import sys 
sys.path.append("..")
import json 
import pdb
import random 
import torch 
import numpy as np 
from torch.utils import data
sys.path.append("../../../")
from utils.utils import EntityMarker


class REDataset(data.Dataset):
    """Data loader for semeval, tacred
    """
    def __init__(self, path, mode, args):
        data = []
        with open(os.path.join(path, mode)) as f:
            all_lines = f.readlines()
            for line in all_lines:
                ins = json.loads(line)
                data.append(ins)
        
            
        entityMarker = EntityMarker(args)
        tot_instance = len(data)

        # load rel2id and type2id
        if os.path.exists(os.path.join(path, "rel2id.json")):
            rel2id = json.load(open(os.path.join(path, "rel2id.json")))
        else:
            raise Exception("Error: There is no `rel2id.json` in "+ path +".")
        if os.path.exists(os.path.join(path, "type2id.json")):
            type2id = json.load(open(os.path.join(path, "type2id.json")))
        else:
            print("Warning: There is no `type2id.json` in "+ path +", If you want to train model using `OT`, `CT` settings, please firstly run `utils.py` to get `type2id.json`.")
    
        print("开始预处理文件 " + mode)
        # pre process data
        self.input_ids = np.zeros((tot_instance, args.max_length), dtype=int)
        self.mask = np.zeros((tot_instance, args.max_length), dtype=int) 
        self.h_pos = np.zeros((tot_instance), dtype=int)
        self.t_pos = np.zeros((tot_instance), dtype=int)
        self.label = np.zeros((tot_instance), dtype=int)
        print_num = 5
        for i, ins in enumerate(data):
            self.label[i] = rel2id[ins["relation"]]            
            # tokenize, 上下文+实体提及
            if args.mode == "CM":
                ids, ph, pt = entityMarker.tokenize(data[i]["token"], data[i]['h']['pos'], data[i]['t']['pos'])
            elif args.mode == "OC":
                ids, ph, pt = entityMarker.tokenize(data[i]["token"], data[i]['h']['pos'], data[i]['t']['pos'], None, None, True, True)
            elif args.mode == "CT":
                h_type = "[unused%d]" % (type2id['subj_'+ins['h']['type']] + 10)
                t_type = "[unused%d]" % (type2id['obj_'+ins['t']['type']] + 10)
                ids, ph, pt = entityMarker.tokenize(data[i]["token"], data[i]['h']['pos'], data[i]['t']['pos'], h_type, t_type)
            elif args.mode == "OM":
                head = entityMarker.tokenizer.tokenize(ins['h']['name'])
                tail = entityMarker.tokenizer.tokenize(ins['t']['name'])
                h_first = ins['h']['pos'][0] < ins['t']['pos'][0]
                ids, ph, pt = entityMarker.tokenize_OMOT(head, tail, h_first)
            elif args.mode == "OT":
                h_type = "[unused%d]" % (type2id['subj_'+ins['h']['type']] + 10)
                t_type = "[unused%d]" % (type2id['obj_'+ins['t']['type']] + 10)
                h_first = ins['h']['pos'][0] < ins['t']['pos'][0]
                ids, ph, pt = entityMarker.tokenize_OMOT([h_type,], [t_type,], h_first)
            else:
                raise Exception("No such mode! Please make sure that `mode` takes the value in {CM,OC,CT,OM,OT}")

            length = min(len(ids), args.max_length)
            self.input_ids[i][0:length] = ids[0:length]
            self.mask[i][0:length] = 1
            self.h_pos[i] = min(ph, args.max_length-1) 
            self.t_pos[i] = min(pt, args.max_length-1)
            #打印前几个元素样本示例
            if i < print_num:
                print(f"打印第{i+1}个样本")
                print(f'样本的tokens {data[i]["token"]}')
                print(f"样本的inputs_ids{self.input_ids[i]}")
                print('输入样本的模式是: .... "[unused0] " + 实体1 + " [unused1] .... [unused2] " + 实体2 + " [unused3]"...')
                print(f"样本的inputs_ids到tokens{entityMarker.tokenizer.convert_ids_to_tokens(self.input_ids[i])}")
                print(f"样本的mask {self.mask[i]}")
                print(f"样本的第一个实体 {data[i]['h']['name']}")
                print(f"样本的第一个实体的位置, 输入到模型中的位置,这个[unused0]位置用来预测关系 {self.h_pos[i]}")
                print(f"样本的第二个实体 {data[i]['t']['name']}")
                print(f"样本的第二个实体的位置,输入到模型中的位置,这个[unused2]位置用来预测关系 {self.t_pos[i]}")
                print(f"样本的的标签labelid {self.label[i]}")
                print(f"样本的的标签label {data[i]['relation']}")
                print()
        print("tokenizer无法找到头/尾实体的句子数量为 %d" % entityMarker.err)
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        mask = self.mask[index]
        h_pos = self.h_pos[index]
        t_pos = self.t_pos[index]
        label = self.label[index]
        # eg: 返回数据示例
        # h_pos = 13
        # index = 244
        # input_ids = [  101  1996     3 23788  5313 23025  2314     4  2003  1037  8914  1997,  1996     1  9610  2226  3567  2314     2  1999  6339  1012   102     0,     0     0     0     0     0     0     0     0     0     0     0     0,     0     0     0     0     0     0     0     0     0     0     0     0,     0     0     0     0     0     0     0     0     0     0     0     0,     0     0     0     0     0     0     0     0     0     0     0     0,     0     0     0     0     0     0     0     0     0     0     0     0,     0     0     0     0     0     0     0     0     0     0     0     0,     0     0     0     0]
        # label = 48
        # mask = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0, 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0, 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        # t_pos = 2
        return input_ids, mask, h_pos, t_pos, label, index
