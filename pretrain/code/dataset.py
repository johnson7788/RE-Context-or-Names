
import json 
import random
import os 
import sys 
sys.path.append("..")
import pdb 
import re 
import pdb 
import math 
import torch
import numpy as np  
from collections import Counter
from torch.utils import data
sys.path.append("../../")
from utils.utils import EntityMarker


class CPDataset(data.Dataset):
    """Overwritten class Dataset for model CP.

    This class prepare data for training of CP.
    """
    def __init__(self, path, args):
        """Inits tokenized sentence and positive pair for CP.
        
        Args:
            path: path to your dataset.
            args: args from command line.
        
        Returns:
            No returns
        
        Raises:
            If the dataset in `path` is not the same format as described in 
            file 'prepare_data.py', there may raise:
                - `key nor found`
                - `integer can't be indexed`
                and so on.
        """
        self.path = path 
        self.args = args 
        data = json.load(open(os.path.join(path, "cpdata.json")))
        rel2scope = json.load(open(os.path.join(path, "rel2scope.json")))
        entityMarker = EntityMarker()

        self.tokens = np.zeros((len(data), args.max_length), dtype=int)
        self.mask = np.zeros((len(data), args.max_length), dtype=int)
        self.label = np.zeros((len(data)), dtype=int)
        self.h_pos = np.zeros((len(data)), dtype=int)
        self.t_pos = np.zeros((len(data)), dtype=int)

        # Distant supervised label for sentence.
        # Sentences whose label are the same in a batch 
        # is positive pair, otherwise negative pair.
        for i, rel in enumerate(rel2scope.keys()):
            scope = rel2scope[rel]
            for j in range(scope[0], scope[1]):
                self.label[j] = i

        for i, sentence in enumerate(data):
            h_flag = random.random() > args.alpha
            t_flag = random.random() > args.alpha
            h_p = sentence["h"]["pos"][0] 
            t_p = sentence["t"]["pos"][0]
            ids, ph, pt = entityMarker.tokenize(sentence["tokens"], [h_p[0], h_p[-1]+1], [t_p[0], t_p[-1]+1], None, None, h_flag, t_flag)
            length = min(len(ids), args.max_length)
            self.tokens[i][:length] = ids[:length]
            self.mask[i][:length] = 1
            self.h_pos[i] = min(args.max_length-1, ph) 
            self.t_pos[i] = min(args.max_length-1, pt)
        print("The number of sentence in which tokenizer can't find head/tail entity is %d" % entityMarker.err)
        # Samples positive pair dynamically. 
        self.__sample__()
    
    def __pos_pair__(self, scope):
        """Generate positive pair.

        Args:
            scope: A scope in which all sentences' label are the same.
                scope example: [0, 12]

        Returns:
            all_pos_pair: All positive pairs. 
            ! IMPORTTANT !
            Given that any sentence pair in scope is positive pair, there
            will be totoally (N-1)N/2 pairs, where N equals scope[1] - scope[0].
            The positive pair's number is proportional to N^2, which will cause 
            instance imbalance. And If we consider all pair, there will be a huge 
            number of positive pairs.
            So we sample positive pair which is proportional to N. And in different epoch,
            we resample sentence pair, i.e. dynamic sampling.
        """
        pos_scope = list(range(scope[0], scope[1]))
        
        # shuffle bag to get different pairs
        random.shuffle(pos_scope)   
        all_pos_pair = []
        bag = []
        for i, index in enumerate(pos_scope):
            bag.append(index)
            if (i+1) % 2 == 0:
                all_pos_pair.append(bag)
                bag = []
        return all_pos_pair
    
    def __sample__(self):
        """Samples positive pairs.

        After sampling, `self.pos_pair` is all pairs sampled.
        `self.pos_pair` example: 
                [
                    [0,2],
                    [1,6],
                    [12,25],
                    ...
                ]
        """
        rel2scope = json.load(open(os.path.join(self.path, "rel2scope.json")))
        self.pos_pair = []
        for rel in rel2scope.keys():
            scope = rel2scope[rel]
            pos_pair = self.__pos_pair__(scope)
            self.pos_pair.extend(pos_pair)

        print("Postive pair's number is %d" % len(self.pos_pair))

    def __len__(self):
        """Number of instances in an epoch.
        
        Overwitten function.
        """
        return len(self.pos_pair)

    def __getitem__(self, index):
        """Get training instance.

        Overwitten function.
        
        Args:
            index: Instance index.
        
        Return:
            input: Tokenized word id.
            mask: Attention mask for bert. 0 means masking, 1 means not masking.
            label: Label for sentence.
            h_pos: Position of head entity.
            t_pos: Position of tail entity.
        """
        bag = self.pos_pair[index]
        input = np.zeros((self.args.max_length * 2), dtype=int)
        mask = np.zeros((self.args.max_length * 2), dtype=int)
        label = np.zeros((2), dtype=int)
        h_pos = np.zeros((2), dtype=int)
        t_pos = np.zeros((2), dtype=int)

        for i, ind in enumerate(bag):
            input[i*self.args.max_length : (i+1)*self.args.max_length] = self.tokens[ind]
            mask[i*self.args.max_length : (i+1)*self.args.max_length] = self.mask[ind]
            label[i] = self.label[ind]
            h_pos[i] = self.h_pos[ind]
            t_pos[i] = self.t_pos[ind]

        return input, mask, label, h_pos, t_pos


class MTBDataset(data.Dataset):
    """Overwritten class Dataset for model MTB.

    该class为训练MTB准备数据。
    """
    def __init__(self, path, args):
        """
        初始化MTB的tokenized sentence和positive pair。
        Args:
            path: path to your dataset.
            args: args from command line.
        
        Returns:
            No returns
        
        Raises:
            If the dataset in `path` is not the same format as described in 
            file 'prepare_data.py', there may raise:
                - `key nor found`
                - `integer can't be indexed`
                and so on.
        """
        self.path = path 
        self.args = args 
        data = json.load(open(os.path.join(path, "mtbdata.json")))
        # 将原始文本转换为BERT输入的ID，并找到实体位置。
        entityMarker = EntityMarker()
        
        # Important Configures, 句子总数
        tot_sentence = len(data)

        # 将token转换为ID，同时将某些实体随机化为“BLANK”。
        # 初始化 tokens, mask , h_pos, t_pos
        self.tokens = np.zeros((tot_sentence, args.max_length), dtype=int)
        self.mask = np.zeros((tot_sentence, args.max_length), dtype=int)
        self.h_pos = np.zeros((tot_sentence), dtype=int)
        self.t_pos = np.zeros((tot_sentence), dtype=int)
        #迭代数据
        for i, sentence in enumerate(data):
            # token被替换成BLANK的概率
            h_flag = random.random() > args.alpha
            t_flag = random.random() > args.alpha
            # 实体1和实体2的位置, eg: h_p: [8, 9, 10, 11]
            h_p = sentence["h"]["pos"][0]  
            t_p = sentence["t"]["pos"][0]
            # 将原始文本转换为BERT输入的ID，并找到实体位置。ids是句子tokenizer后的id，ph是 头实体位置(头实体标记的起始位置)。这里是[unused0]的位置，
            # pt 是尾部实体位置(尾部实体标记的起始位置)。这里是[unused2]的位置
            ids, ph, pt = entityMarker.tokenize(sentence["tokens"], [h_p[0], h_p[-1]+1], [t_p[0], t_p[-1]+1], None, None, h_flag, t_flag)
            # 为了计算mask，
            length = min(len(ids), args.max_length)
            #把默认为0的token替换成实际的tokenid，其余部分相当于padding了
            self.tokens[i][0:length] = ids[0:length]
            # 只有对应有tokenid的位置时1，其它默认为0了
            self.mask[i][0:length] = 1
            #表明第一个实体的位置，
            self.h_pos[i] = min(args.max_length-1, ph)
            self.t_pos[i] = min(args.max_length-1, pt)
        print("tokenizer找不到头/尾实体的句子数%d" % entityMarker.err)

        entpair2scope = json.load(open(os.path.join(path, "entpair2scope.json")))
        entpair2negpair = json.load(open(os.path.join(path, "entpair2negpair.json")))
        self.pos_pair = []
        
        # 生成所有正样本对。 eg: self.pos_pair: [[0, 1], [2, 3], [4, 5]]
        for key in entpair2scope.keys():
            self.pos_pair.extend(self.__pos_pair__(entpair2scope[key]))
        print("Positive pairs 数量是 %d" % len(self.pos_pair))
        # 负样本动态采样
        self.__sample__()

    def __sample__(self):    
        """Sample hard negative pairs.

        Sample hard negative pairs for MTB. As described in `prepare_data.py`, 
        `entpair2negpair` is ` A python dict whose key is `head_id#tail_id`. And the value
                is the same format as key, but head_id or tail_id is different(only one id is 
                different). ` 
        
        ! IMPORTANT !
        We firstly get all hard negative pairs which may be a very huge number and then we sam
        ple negaitive pair where sampling number equals positive pairs' numebr. Using our 
        dataset, this code snippet can run normally. But if your own dataset is very big, this 
        code snippet will cost a lot of memory.
        """
        entpair2scope = json.load(open(os.path.join(self.path, "entpair2scope.json")))
        entpair2negpair = json.load(open(os.path.join(self.path, "entpair2negpair.json")))
        neg_pair = []

        # Gets all negative pairs.
        for key in entpair2negpair.keys():
            my_scope = entpair2scope[key]
            entpairs = entpair2negpair[key]
            if len(entpairs) == 0:
                continue
            for entpair in entpairs:
                neg_scope = entpair2scope[entpair]
                neg_pair.extend(self.__neg_pair__(my_scope, neg_scope))
        print("(MTB)Negative pairs number is %d" %len(neg_pair))
        
        # Samples a same number of negative pair with positive pairs. 
        random.shuffle(neg_pair)
        self.neg_pair = neg_pair[0:len(self.pos_pair)]
        del neg_pair # save the memory 

    def __pos_pair__(self, scope):
        """
        获取所有正样本对。
        Args:
            scope: 所有句子共享相同的实体对(头部实体和尾部实体)的范围。  eg: [0, 2]   return: [[0, 1]]
            eg: scope: [2, 4] ; pos_pair:[[2, 3]]
            eg: [4, 6] --> [[4, 5]]
        Returns:
            pos_pair：scope中的所有positive pairs。
            scope中的positive pairs为(N-1)N/2，其中N等于scope[1] - scope[0]
        """
        ent_scope = list(range(scope[0], scope[1]))
        pos_pair = []
        for i in range(len(ent_scope)):
            for j in range(i+1, len(ent_scope)):
                pos_pair.append([ent_scope[i], ent_scope[j]])
        return pos_pair

    def __neg_pair__(self, my_scope, neg_scope):
        """Gets all negative pairs in different scope.

        Args:
            my_scope: A scope which is samling negative pairs.
            neg_scope: A scope where sentences share only one entity
                with sentences in my_scope.
        
        Returns:
            neg_pair: All negative pair. Sentences in different scope 
                make up negative pairs.
        """
        my_scope = list(range(my_scope[0], my_scope[1]))
        neg_scope = list(range(neg_scope[0], neg_scope[1]))
        neg_pair = []
        for i in my_scope:
            for j in neg_scope:
                neg_pair.append([i, j])
        return neg_pair

    def __len__(self):
        """Number of instances in an epoch.
        
        Overwitten function.
        """
        return len(self.pos_pair)

    def __getitem__(self, index):
        """Gets training instance.

        If index is odd, we will return nagative instance, otherwise 
        positive instance. So in a batch, the number of positive pairs 
        equal the number of negative pairs.

        Args:
            index: Data index.
        
        Returns:
            {l,h}_input: Tokenized word id.
            {l,h}_mask: Attention mask for bert. 0 means masking, 1 means not masking.
            {l,h}_ph: Position of head entity.
            {l,h}_pt: Position of tail entity.
            label: Positive or negative.

            Setences in the same position in l_input and r_input is a sentence pair
            (positive or negative).
        """
        if index % 2 == 0:
            l_ind = self.pos_pair[index][0]
            r_ind = self.pos_pair[index][1]
            label = 1
        else:
            l_ind = self.neg_pair[index][0]
            r_ind = self.neg_pair[index][1]
            label = 0
        
        l_input = self.tokens[l_ind]
        l_mask = self.mask[l_ind]
        l_ph = self.h_pos[l_ind]
        l_pt = self.t_pos[l_ind]
        r_input = self.tokens[r_ind]
        r_mask = self.mask[r_ind]
        r_ph = self.h_pos[r_ind]
        r_pt = self.t_pos[r_ind]

        return l_input, l_mask, l_ph, l_pt, r_input, r_mask, r_ph, r_pt, label

