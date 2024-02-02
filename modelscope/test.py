import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from modelscope import (
    snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)

import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
model_id = 'qwen/Qwen-VL-Chat'
revision = 'v1.1.0'

# model_dir = snapshot_download(model_id, revision=revision)
model_dir = "model/qwen/Qwen-VL-Chat"  # 本地加载
torch.manual_seed(1234)

# 请注意：分词器默认行为已更改为默认关闭特殊token攻击防护。
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, bf16=True).eval()
# 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, fp16=True).eval()
# 使用CPU进行推理，需要约32GB内存
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cpu", trust_remote_code=True).eval()
# 默认使用自动模式，根据设备自动选择精度
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()

# 可指定不同的生成长度、top_p等相关超参
model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)

# 第一轮对话 1st dialogue turn

query = tokenizer.from_list_format([
    # {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
    {'image': "data/test.png"},
    {'text': '这是什么'},
])

response, history = model.chat(tokenizer, query=query, history=None)
print(response)
# 图中是一名年轻女子在沙滩上和她的狗玩耍，狗的品种是拉布拉多。她们坐在沙滩上，狗的前腿抬起来，与人互动。

# 第二轮对话 2st dialogue turn
response, history = model.chat(tokenizer, '可以说出这个设备是什么型号吗', history=history)
print(response)
