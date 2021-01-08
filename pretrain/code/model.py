import os 
import pdb 
import torch
import torch.nn as nn 
from pytorch_metric_learning.losses import NTXentLoss
from transformers import BertForMaskedLM, BertForPreTraining, BertTokenizer

def mask_tokens(inputs, tokenizer, not_mask_pos=None):
    """准备用于mask语言模型的mask token inputs/label, 15%的token中的 80% MASK, 10% random, 10% original.
    
    Args:
        inputs: Inputs to mask. (batch_size, max_length) 
        tokenizer: Tokenizer.
        not_mask_pos: 避免mask实体提及，1表示不要mask, torch.Size([32, 64])
    
    Returns:
        inputs: Masked inputs.
        labels: Masked language model labels.
    """

    if tokenizer.mask_token is None:
        raise ValueError(
            "该tokenizer没有mask token，这是mask语言模型所必需的。 如果要使用此tokenizer，请删除--mlm标志。"
        )
    # 准备一个labels，初始化，其实labels就是开始输入的inputs， 开始的inputs被mask后作为新的inputs
    labels = inputs.clone()
    # 我们在每个序列中对masked-LM 训练采样了一些token(在Bert / RoBERTa中args.mlm_probability的默认值为0.15)
    # 制作一个形式为(batch_size, max_length) 的，用0.15填充的矩阵
    probability_matrix = torch.full(labels.shape, 0.15)
    # 获取specialtoken的位置，eg，其中的一个元素, 这里的1一般是CLS和SEP [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    #special token的位置，直接赋值为0.0, 其余位置还是0.15
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        # tokenizer.pad_token_id是等于0，labels等于0的地方表示是padding的部分， eg: 其中一个元素，这里是一个句子 tensor([False, False, False, False, False, False, False, False, False, False,
        #         False, False, False, False, False, False, False, False, False, False,
        #         False, False, False, False, False, False, False, False, False, False,
        #         False, False, False, False, False, False, False,  True,  True,  True,
        #          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #          True,  True,  True,  True])
        padding_mask = labels.eq(tokenizer.pad_token_id)
        # padding的部分也赋值为0.0
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    if not_mask_pos is None:
        masked_indices = torch.bernoulli(probability_matrix).bool()
    else:
        # 15%的概率这个位置会被设为0，not_mask_pos取反后，与操作，表示这个位置是实体位置，不能被mask掉
        masked_indices = torch.bernoulli(probability_matrix).bool() & (~(not_mask_pos.bool())) # ** can't mask entity marker **
    #我们只计算mask的token的位置的损失, masked_indices取反，masked_indices: tensor([[False, False, False,  ..., False, False, False],...)
    labels[~masked_indices] = -100  #
    # labels 其中一个元素 pd.DataFrame(labels.numpy()[0]), 只计算不是位置为-100处的损失 [0: -100], [1: -100], [2: -100], [3: -100], [4: -100], [5: -100], [6: -100], [7: -100], [8: -100], [9: -100], [10: -100], [11: -100], [12: -100], [13: -100], [14: -100], [15: -100], [16: -100], [17: -100], [18: -100], [19: -100], [20: -100], [21: -100],
    # 其中要被mask的token中 80% 的几率, 我们替换这个token为 [MASK] tokenizer.mask_token, 这个表示我们取的15%的mask的token中，80%的几率被mask替换
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    # 替换inputs中的位置
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    #生成形状为labels.shape,  数为字典长度中的随机的数字，作为随机的单词
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    # 替换inputs中的位置
    inputs[indices_random] = random_words[indices_random]

    # 剩余10%的几率保持不变 The rest of the time (10% of the time) we keep the masked input tokens unchanged
    # return inputs.cuda(), labels.cuda()
    return inputs, labels

class CP(nn.Module):
    """Contrastive Pre-training model.

    This class implements `CP` model based on model `BertForMaskedLM`. And we 
    use NTXentLoss as contrastive loss function.

    Attributes:
        model: Model to train.
        tokenizer: Tokenizer.
        ntxloss: Contrastive loss function.
        args: Args from command line. 
    """
    def __init__(self, args):
        super(CP, self).__init__()
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.ntxloss = NTXentLoss(temperature=args.temperature)
        self.args = args 
    
    def forward(self, input, mask, label, h_pos, t_pos):
        # masked language model loss
        input = input.view(-1, self.args.max_length)
        mask = mask.view(-1, self.args.max_length)
        label = label.view(-1) # (batch_size * 2)
        h_pos = h_pos.view(-1)
        t_pos = t_pos.view(-1)

        # Ensure that `mask_tokens` function doesn't mask entity mention.
        indice = torch.arange(0, input.size()[0])
        not_mask_pos = torch.zeros((input.size()[0], input.size()[1]), dtype=int)
        not_mask_pos[indice, h_pos] = 1
        not_mask_pos[indice, t_pos] = 1

        m_input, m_labels = mask_tokens(input.cpu(), self.tokenizer, not_mask_pos)
        m_outputs = self.model(input_ids=m_input, labels=m_labels, attention_mask=mask)
        m_loss = m_outputs[1]

        outputs = m_outputs

        # entity marker starter
        batch_size = input.size()[0]
        indice = torch.arange(0, batch_size)
        h_state = outputs[0][indice, h_pos] # (batch_size * 2, hidden_size)
        t_state = outputs[0][indice, t_pos]
        state = torch.cat((h_state, t_state), 1)

        r_loss = self.ntxloss(state, label)

        return m_loss, r_loss



class MTB(nn.Module):
    """Matching the Blanks.

    此class基于模型`BertForMaskedLM` 实现`MTB`模型。

    Attributes:
        model: Model to train.
        tokenizer: Tokenizer.
        bceloss: Binary Cross Entropy loss.
    """
    def __init__(self, args):
        super(MTB, self).__init__()
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # 加了sigmoid 的BCELoss损失, 就是内部自动计算logit, 就是逻辑回归损失，Binary Cross Entropy, 或者二分类的交叉熵损失
        self.bceloss = nn.BCEWithLogitsLoss()
        self.args = args
    

    def forward(self, l_input, l_mask, l_ph, l_pt, r_input, r_mask, r_ph, r_pt, label):
        """
        样本来自dataset.py中的MTBDataset的函数__getitem__， 一个批次的数据
        compute not mask entity marker
        :param l_input: torch.Size([32, 64]), [batch_size, seq_len] 样本对的第一个样本，左侧样本，或者称为样本1，token id
        :param l_mask: torch.Size([32, 64]),  [batch_size, seq_len]  token 的mask值, 0,1组成的序列
        :param l_ph: l_ph 代表左侧样本的头实体的位置，头实体，即第一个实体
        :param l_pt: l_ph 代表左侧样本的尾实体的位置，尾实体，即第二个实体
        :param r_input: 第二个样本的 input id
        :param r_mask: 第二个样本
        :param r_ph: [batch_size]
        :param r_pt:  [batch_size]
        :param label: [batch_size], 这里只是分正样本和负样本，所以是0，1组成
        :return:
        """
        indice = torch.arange(0, l_input.size()[0])
        # l_not_mask_pos和r_not_mask_pos是全0的tensor，和输入的形状一致
        l_not_mask_pos = torch.zeros((l_input.size()[0], l_input.size()[1]), dtype=int) 
        r_not_mask_pos = torch.zeros((l_input.size()[0], l_input.size()[1]), dtype=int) 

        # 确保 mask_tokens 函数不mask实体提及, 只有l_ph和l_pt的位置为1，其它的位置为0
        l_not_mask_pos[indice, l_ph] = 1
        l_not_mask_pos[indice, l_pt] = 1

        r_not_mask_pos[indice, r_ph] = 1
        r_not_mask_pos[indice, r_pt] = 1

        # masked language model loss, 语言模型的mask操作，和BERT一样，但是加入了不要mask实体位置信息
        m_l_input, m_l_labels = mask_tokens(l_input.cpu(), self.tokenizer, l_not_mask_pos)
        m_r_input, m_r_labels = mask_tokens(r_input.cpu(), self.tokenizer, r_not_mask_pos)
        #是否使用GPU
        if self.args.gpu:
            m_l_input = m_l_input.cuda()
            m_l_labels = m_l_labels.cuda()
            m_r_input = m_r_input.cuda()
            m_r_labels = m_r_labels.cuda()

        m_l_outputs = self.model(input_ids=m_l_input, labels=m_l_labels, attention_mask=l_mask)
        m_r_outputs = self.model(input_ids=m_r_input, labels=m_r_labels, attention_mask=r_mask)
        m_loss = m_l_outputs[1] + m_r_outputs[1]

        # sentence pair relation loss 
        l_outputs = m_l_outputs
        r_outputs = m_r_outputs

        batch_size = l_input.size()[0]
        indice = torch.arange(0, batch_size)
        
        # left output
        l_h_state = l_outputs[0][indice, l_ph] # (batch, hidden_size)
        l_t_state = l_outputs[0][indice, l_pt] # (batch, hidden_size)
        l_state = torch.cat((l_h_state, l_t_state), 1) # (batch, 2 * hidden_size)
        
        # right output 
        r_h_state = r_outputs[0][indice, r_ph] 
        r_t_state = r_outputs[0][indice, r_pt]
        r_state = torch.cat((r_h_state, r_t_state), 1)

        # cal similarity
        similarity = torch.sum(l_state * r_state, 1) # (batch)

        # cal loss
        r_loss = self.bceloss(similarity, label.float())

        return m_loss, r_loss 
        


