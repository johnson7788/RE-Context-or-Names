此目录包含用于预训练的代码和数据。

### 1. Dataset 

您可以从[google drive](https://drive.google.com/file/d/1V9C8678G-zudBa2rzFtH1ZeNieh3UFG_/view?usp=sharing) 
或[Tsinghua cloud](https://cloud.tsinghua.edu.cn/f/f55fd09903c94baa9436/?dl=1)。 
然后将数据集放在“./data”目录中。 (您可能需要`mkdir data`)，然后运行`code/prepare_data.py`准备预训练数据。


### 2. Pretrained Model

您可以从以下位置下载我们的预训练模型

MTB预训练模型：  [google drive](https://drive.google.com/file/d/1viGnWGg3B-LasR9UWQFg3lhl-hOEl5ed/view?usp=sharing) 
or [Tsinghua cloud](https://cloud.tsinghua.edu.cn/f/5ce773cc67294ce488e5/?dl=1), 

CP预训练模型: [google drive](https://drive.google.com/file/d/1WU39lYAkZ9JYXlCZFGyAxBlQ--5IU4m6/view?usp=sharing) 
or [Tsinghua cloud](https://cloud.tsinghua.edu.cn/f/4097d1055962483cb6d9/?dl=1). 
然后把他们放到 `./ckpt` 目录中.(You may need `mkdir ckpt`).

### 3. Pretrain
预训练MTB的方法:
```shell
python -m torch.distributed.launch --nproc_per_node 4  main.py \
	--cuda 4,5,6,7 \
	--model MTB \
	--lr 3e-5 --batch_size_per_gpu 32 --max_epoch 20 \
	--gradient_accumulation_steps 2 \
	--max_length 64 \
	--save_step 5000 \
	--alpha 0.3 \
	--train_sample \
	--save_dir ckpt_mtb \
```
预训练CP的方法:

```shell
python -m torch.distributed.launch --nproc_per_node 4  main.py \
	--cuda 4,5,6,7 \
	--model CP \
	--lr 3e-5 --batch_size_per_gpu 32 --max_epoch 20 \
	--gradient_accumulation_steps 16 \
	--max_length 64 \
	--save_step 500 \
	--alpha 0.3 \
	--temperature 0.05 \
	--train_sample \
	--save_dir ckpt_cp \
```
