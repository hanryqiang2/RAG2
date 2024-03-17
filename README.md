# RAG2
准备QA数据集和知识文档
准备QA数据集：
QA数据集通过预处理保存在文件夹内，source每行存一个问题，target每行存一个答案。
QA文件名字一定要叫做train.source、train.target、test.source、test.target、val.source、val.target
准备知识文档：
知识文档先通过预处理保存为一个以制表符\t分隔的csv文件，不需要列名，第一列是title，第二列是text。
运行use_own_knowledge_dataset.py即可对知识文档编码并创建索引。(记得修改自己想用的文档编码器和保存路径）

# 微调启动
使用bash文件微调RAG
bash finetune_rag_ray.sh

重要参数：
data_dir  QA数据文件夹地址
output_dir  微调后模型保存地址
model_name_or_path 需要微调的RAG模型地址
train_batch_size  训练时batch_size大小
eval_batch_size  验证时batch_size大小
max_source_length QA中问题的最大长度
max_target_length QA中答案的最大长度
val_max_target_length 
test_max_target_length 
num_train_epochs  训练轮次
passages_path 知识文档地址
index_path  知识文档索引地址

# 如需冻结某个组件：
1、finetune_rag.py  199-201行的注释部分打开，即开始冻结，如果冻结问题编码器，就：
for param in self.model.question_encoder.parameters():
	param.requires_grad = False
    如果冻结生成器，就：
for param in self.model.generator.parameters():
	param.requires_grad = False
2、lightning_base.py  146-157行注释打开，134-144行注释掉，即开始冻结，如果是冻结问题编码器，引号内就填“question_encoder”；如果是冻结生成器，引号内就填“generator”
以上两点同时修改，才能冻结某个组件

# 微调后保存模型策略：
call_backs_rag.py 37-44行 every_n_epochs 设置多少，就每隔几轮保存一次，目前设定的是以loss为判断，如果验证集上loss不断上升，就只会保存最开始k个模型，k由save_top_k决定。

训练完后查看metric：
train中保存的loss和设置的train_batch_size有关，需要在finetune_rag.py的321行修改，比如batch_size是10，就：
avg_train_metrics = {name: (torch.stack([x["log"][name] for x in outputs]).mean()/10) for name in self.loss_names}
