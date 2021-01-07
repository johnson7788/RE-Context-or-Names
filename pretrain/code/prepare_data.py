import json 
import random
import os 
import sys 
import pdb 
import re 
import torch
import argparse
import numpy as np 
from tqdm import trange
from collections import Counter, defaultdict


def filter_sentence(sentence):
    """单个句子过滤，Filter sentence.
    
    Filter sentence:
        - head mention equals tail mention
        - head mentioin and tail mention overlap

    Args:
        sentence: A python dict.
            sentence example:
            {
                'tokens': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                'r': 'P1'
            }

    Returns:
         True or False。 如果句子中包含上述异常条件，则返回True。 其他返回False

    Raises:
        如果句子的格式与上述格式不同，
        则此函数可能会因Python Interpreter引发“key not found”错误。
    """
    head_pos = sentence["h"]["pos"][0]
    tail_pos = sentence["t"]["pos"][0]
    
    if sentence["h"]["name"] == sentence["t"]["name"]:  # head mention equals tail mention
        return True

    if head_pos[0] >= tail_pos[0] and head_pos[0] <= tail_pos[-1]: # head mentioin and tail mention overlap
        return True
    
    if tail_pos[0] >= head_pos[0] and tail_pos[0] <= head_pos[-1]: # head mentioin and tail mention overlap
        return True  

    return False


def process_data_for_CP(data):
    """Process data for CP. 

    This function will filter NA relation, abnormal sentences,
    and relation of which sentence number is less than 2(This relation
    can't form positive sentence pair).

    Args:
        data: Original data for pre-training and is a dict whose key is relation.
            data example:
                {
                    'P1': [
                        {
                            'tokens': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                            'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                            't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                            'r': 'P1'
                        },
                        ...
                    ],
                    ...
                }

    Returns: 
        No returns. 
        But this function will save two json-formatted files:
            - list_data: A list of sentences.
            - rel2scope: A python dict whose key is relation and value is 
                a scope which is left-closed-right-open `[)`. All sentences 
                in a same scope share the same relation.
            
            example:
                - list_data:
                    [
                        {
                            'tokens': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                            'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                            't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                            'r': 'P1'
                        },
                        ...
                    ]
                
                - rel2scope:
                    {
                        'P10': [0, 233],
                        'P1212': [233, 1000],
                        ....
                    }
        
    Raises:
        If data's format isn't the same as described above, 
        this function may raise `key not found` error by Python Interpreter.
    """
    washed_data = {}
    for key in data.keys():
        if key == "P0":
            continue
        rel_sentence_list = []
        for sen in data[key]:
            if filter_sentence(sen):
                continue
            rel_sentence_list.append(sen)
        if len(rel_sentence_list) < 2:
            continue        
        washed_data[key] = rel_sentence_list

    ll = 0
    rel2scope = {}
    list_data = []
    for key in washed_data.keys():
        list_data.extend(washed_data[key])
        rel2scope[key] = [ll, len(list_data)]
        ll = len(list_data)
    
    if not os.path.exists("../data/CP"):
        os.mkdir("../data/CP")
    json.dump(list_data, open("../data/CP/cpdata.json","w"))
    json.dump(rel2scope, open("../data/CP/rel2scope.json", 'w'))
    print(f"CP 数据处理完成")
    os.system('ls ../data/CP')


def process_data_for_MTB(data):
    """Process data for MTB. 

    此函数将过滤异常句子，以及句子数小于2的实体对(该实体对不能形成positive句子对)。

    Args:
        data: Original data for pre-training and is a dict whose key is relation.
            data example:
                {
                    'P1': [
                        {
                            'token': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                            'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                            't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                            'r': 'P1'
                        },
                        ...
                    ],
                    ...
                }

    Returns: 
        No returns. 
        但是此功能将保存三个json格式的文件 :
            - list_data: A list of sentences.
            - entpair2scope: A python dict whose key is `head_id#tail_id` and value is 
                a scope which is left-closed-right-open `[)`. All sentences in one same 
                scope share the same entity pair
            - entpair2negpair: A python dict whose key is `head_id#tail_id`. And the value
                is the same format as key, but head_id or tail_id is different(only one id is 
                different). 

            example:
                - list_data:
                    [
                        {
                            'tokens': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                            'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                            't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                            'r': 'P1'
                        },
                        ...
                    ]
                - entpair2scope:
                    {
                        'Q1234#Q2356': [0, 233],
                        'Q135656#Q10': [233, 1000],
                        ....
                    }
                - entpair2negpair:
                    {
                        'Q1234#Q2356': ['Q1234#Q3560','Q923#Q2356', 'Q1234#Q100'],
                        'Q135656#Q10': ['Q135656#Q9', 'Q135656#Q10010', 'Q2666#Q10']
                    }
        
    Raises:
        如果数据格式与上述格式不同，则此函数可能会因Python Interpreter引发“key not found”错误。
    """
    # 共享同一实体对的最大句子数。设置此参数是为了限制偏向具有很多句子的流行实体对。
    # 当然，您可以更改此参数，但是在我们的实验中，我们使用8。
    max_num = 8

    # 我们更改原始数据的格式。
    # ent_data是一个python字典，其中的key是`head_id#tail_id`，value是包含相同实体对的句子。
    ent_data = defaultdict(list)
    for key in data.keys():
        for sentence in data[key]:
            if filter_sentence(sentence):
                continue
            head = sentence["h"]["id"]
            tail = sentence["t"]["id"]
            ent_data[head + "#" + tail].append(sentence)

    ll = 0
    list_data = []
    entpair2scope = {}
    for key in ent_data.keys():
        if len(ent_data[key]) < 2:
            continue
        list_data.extend(ent_data[key][0:max_num])
        entpair2scope[key] = [ll, len(list_data)]
        ll = len(list_data)
    # 我们将预先生成“硬”负样本。 entpair2negpair是python字典，其key为`head_id#tail_id`。
    # dict的value与key的格式相同，但是head_id或tail_id不同(仅一个id不同)。
    entpair2negpair = defaultdict(list)
    entpairs = list(entpair2scope.keys())
    entpairs.sort(key=lambda a: a.split("#")[0])
    for i in range(len(entpairs)):
        head = entpairs[i].split("#")[0]
        for j in range(i+1, len(entpairs)):
            if entpairs[j].split("#")[0] != head:
                break
            entpair2negpair[entpairs[i]].append(entpairs[j])

    entpairs.sort(key=lambda a: a.split("#")[1])
    for i in range(len(entpairs)):
        tail = entpairs[i].split("#")[1]
        for j in range(i+1, len(entpairs)):
            if entpairs[j].split("#")[1] != tail:
                break
            entpair2negpair[entpairs[i]].append(entpairs[j])

    if not os.path.exists("../data/MTB"):
        os.mkdir("../data/MTB")
    json.dump(entpair2negpair, open("../data/MTB/entpair2negpair.json","w"))
    json.dump(entpair2scope, open("../data/MTB/entpair2scope.json", "w"))
    json.dump(list_data, open("../data/MTB/mtbdata.json", "w"))
    print(f"MTB 数据处理完成:")
    os.system('ls ../data/MTB')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def part_of_dict(d, num, rnd=False):
    """
    返回传入的字典的一部分，字典的一个子集，
    :param d: 字典，
    :param num: 返回字典中的几个元素
    :param rnd: True是表示随机返回，False是按字典顺序返回
    :return: dict
    """
    if rnd:
        import random
        all_keys = list(d.keys())
        rnd_keys = random.sample(all_keys, num)
        newdict = {key:d[key] for key in rnd_keys}
    else:
        iterator = iter(d.items())
        newdict = dict(next(iterator) for i in range(num))
    return newdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据集预处理")
    parser.add_argument("--dataset", dest="dataset", type=str, default="MTB", help="{MTB,CP}")
    args = parser.parse_args()
    set_seed(42)

    data = json.load(open("../data/exclude_fewrel_distant.json"))
    if args.dataset == "CP":
        process_data_for_CP(data)
    elif args.dataset == "MTB":
        process_data_for_MTB(data)
