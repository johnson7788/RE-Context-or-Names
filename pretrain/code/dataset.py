
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
        """
        采样hard负样本对
        为MTB模型采样负样本对. 如 `prepare_data.py`脚本中函数
        `entpair2negpair` 描述,  格式是 `head_id#tail_id`.
        value与key的格式相同，但是head_id或tail_id不同(仅id不同)。

        ! IMPORTANT !
        我们首先得到所有可能得到大量的hard negative pairs，
        然后对egaitive pair进行采样，其中采样数等于positive pairs数。
        使用我们的数据集，此代码段可以正常运行。
         但是，如果您自己的数据集很大，那么此代码段将占用大量内存。
        """
        entpair2scope = json.load(open(os.path.join(self.path, "entpair2scope.json")))
        entpair2negpair = json.load(open(os.path.join(self.path, "entpair2negpair.json")))
        neg_pair = []

        # Gets all negative pairs.
        for key in entpair2negpair.keys():
            # eg: key:'Q1044963#Q1020776', my_scope: [2474, 2478]
            my_scope = entpair2scope[key]
            #  entpairs: ['Q1044963#Q124739']
            entpairs = entpair2negpair[key]
            if len(entpairs) == 0:
                continue
            for entpair in entpairs:
                # neg_scope: [2839, 2845]
                neg_scope = entpair2scope[entpair]
                neg_pair.extend(self.__neg_pair__(my_scope, neg_scope))
        print("(MTB)Negative pairs 数量是 %d" %len(neg_pair))
        
        # 采样相同数量的负对和正对。
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
        """获取不同范围内的所有负对。

        Args:
            my_scope: A scope which is samling negative pairs. eg: [2474, 2478]
            neg_scope: 句子仅与my_scope中的句子共享一个实体的范围。   eg: [2839, 2845]

        
        Returns:
            neg_pair: 所有negative pair。 不同范围的句子组成negative pair。
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
        """获取训练的样本.

        如果index为奇数，我们将返回nagative instance，
        否则将返回positive instance。
        因此，批次中positive pairs的数量等于negative pairs的数量。

        Args:
            index: Data index.
        
        Returns:
            {l,h}_input: Tokenized word id.  size: 64, seq_length
            {l,h}_mask: Attention mask for bert. 0 means masking, 1 means not masking.
            {l,h}_ph: 头实体的位置. eg: l_ph: 18    r_ph:31
            {l,h}_pt: 尾实体的位置 eg: l_pt: 26   r_pt:38
            label: Positive or negative.

            l_input和r_input中处于相同位置是一个句子对(positive or negative)。
        """
        #奇数取负样本，偶数取正样本
        if index % 2 == 0:
            l_ind = self.pos_pair[index][0]
            r_ind = self.pos_pair[index][1]
            label = 1
        else:
            l_ind = self.neg_pair[index][0]
            r_ind = self.neg_pair[index][1]
            label = 0
        #一次取出2个样本，组成样本对，l_input代表左侧的样本，r_input代表右侧的样本，组成一个样本对
        # 奇数取负样本对，偶数取正样本对
        # l_ph 代表左侧样本的头实体的位置
        l_input = self.tokens[l_ind]
        l_mask = self.mask[l_ind]
        l_ph = self.h_pos[l_ind]
        l_pt = self.t_pos[l_ind]
        r_input = self.tokens[r_ind]
        r_mask = self.mask[r_ind]
        r_ph = self.h_pos[r_ind]
        r_pt = self.t_pos[r_ind]

        return l_input, l_mask, l_ph, l_pt, r_input, r_mask, r_ph, r_pt, label

