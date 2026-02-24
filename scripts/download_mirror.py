# 环境变量设为镜像站
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import pipeline

# 使用镜像源下载模型
model_name = "microsoft/deberta-v3-base"

print("正在使用镜像源下载模型...")
pipe = pipeline("fill-mask", model=model_name)

# 测试推理
print("模型加载完成，开始测试推理...")
result = pipe("The capital of France is [MASK].")

print("推理结果:")
print(result)
