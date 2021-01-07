此repo包含下游任务(有监督的RE和Fewshot RE)的代码和数据。 

### 1. Supervised RE

#### 1.1 Dataset

我们在`data/`中提供SemEval，Wiki80，ChemProt，但您需要手动从[LDC](https://catalog.ldc.upenn.edu/LDC2018T24)下载TACRED。 

请确保每个数据集都具有“train.txt”，“dev.txt”，“test.txt”和“rel2id.json”(**如果该基准具有NA关系，则NA必须为0 **)。 
并且`train.txt`，`dev.txt`，`text.txt`，应该有多行，每行具有以下json格式：

```python
{
    "tokens":["Microsoft", "was", "founded", "by", "Bill", "Gates", "."], 
    "h":{
        "name": "Microsotf", "pos":[0,1]  # Left closed and right open interval
    }
    "t":{
        "name": "Bill Gates", "pos":[4,6] # Left closed and right open interval
    }
    "relation": "founded_by"
}
```

**IMPORTANT**

我们不会使用自己的代码来评估SemEval上的模型，而是使用“官方”评估脚本。 
参见https://github.com/sahitya0000/Relation-Classification/tree/master/corpus/SemEval2010_task8_scorer-v1.2
因此，如果SemEval上的结果异常，请使用官方脚本。 可以使用此代码正常评估其他数据集。

#### 1.2 Train

运行以下脚本 :

```shell
bash run.sh
```

如果要使用其他模型，可以在run.sh中更改ckpt 

```shell
array=(42 43 44 45 46)
ckpt="None"
for seed in ${array[@]}
do
	bash train.sh 1 $seed $ckpt 1 6
done
```

"None" 代表 Bert. 您可以在`../pretrain/ckpt`目录中使用任何checkpoint进行微调

#### 2. FewShot RE

我们已经下载了  [FewRel](https://github.com/thunlp/FewRel) into `fewshotRE` 并修改了一些行 .

运行以下脚本： 

```shell
bash run.sh
```

如果要使用其他模型，可以在run.sh中更改ckpt 

```shell
array=(42)
path="paht/to/ckpt"
for seed in ${array[@]}
do
	bash train.sh 7 $seed $path
done
```

"None" 代表 Bert. 您可以在`../pretrain/ckpt`目录中使用任何checkpoint进行微调

**IMPORTANT**

我们不提供测试集。 如果要测试模型，请将结果上传到  https://thunlp.github.io/fewrel.html. See https://github.com/thunlp/FewRel. 