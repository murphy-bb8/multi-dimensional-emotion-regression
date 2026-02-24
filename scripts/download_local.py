# scp上传到远程后调用
from transformers import pipeline

model_path = "/root/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
pipe = pipeline("fill-mask", model=model_path)

# 测试推理
result = pipe("The capital of France is [MASK].")

print(result)