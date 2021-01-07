# RE Analysis

Dataset and code for [Learning from Context or Names? An Empirical Study on Neural Relation Extraction](https://arxiv.org/abs/2010.01923). 

If you use this code, please cite us
```
@article{peng2020learning,
  title={Learning from Context or Names? An Empirical Study on Neural Relation Extraction},
  author={Peng, Hao and Gao, Tianyu and Han, Xu and Lin, Yankai and Li, Peng and Liu, Zhiyuan and Sun, Maosong and Zhou, Jie},
  journal={arXiv preprint arXiv:2010.01923},
  year={2020}
}
```

### Quick Start

您可以按照以下步快速运行我们的代码： 

- 如以下部分所述安装依赖项。 
- cd进入`pretrain` or `finetune`”目录，然后下载并预处理数据以进行预训练或微调。 

### 1. Dependencies

运行以下脚本以安装依赖项。 

```shell
pip install -r requirement.txt
```

**You need install transformers and apex manually.**

**transformers**
我们使用huggingface transformers来实现Bert。 
为了方便起见，我们将[transformers](https://github.com/huggingface/transformers)下载到了utils/中。 
并且我们还修改了src/transformers/modeling_bert.py中的类BertForMaskedLM中的某些行，同时保持其他代码不变。 

您只需要手动安装transformers即可。

```shell
cd utils/transformers  && pip install .
```

**apex**
Install [apex](https://github.com/NVIDIA/apex) under the offical guidance.

### 2. More details
您可以进入进行`pretrain` or `finetune`，以了解有关预训练或微调的更多详细信息。







