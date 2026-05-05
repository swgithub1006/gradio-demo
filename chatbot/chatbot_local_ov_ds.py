import gradio as gr
import openvino_genai as ov_genai
import os

# https://chat.deepseek.com/a/chat/s/d1eb8a9c-60bc-4fd6-ad50-17cd74b08b87
# 1. 加载本地OpenVINO模型
#    请确保此路径下包含 model.xml 和 model.bin 文件。
# model_path = "E:\\models\\OpenVINO\\DeepSeek-R1-Distill-Qwen-7B-int4-ov"
MODEL_PATH = "E:\\models\\OpenVINO\\Qwen3-1.7B-int8-ov"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"错误：找不到本地模型路径 '{model_path}'。请确保路径正确。")

print(f"正在加载模型: {model_path}")
pipe = ov_genai.LLMPipeline(model_path, "CPU")
print("模型加载成功！")

# 2. 定义聊天核心函数
def chat_with_model(message, history):
    # 构建上下文提示词 (可根据你的模型调整对话模板)
    prompt = ""
    for user_msg, bot_msg in history[-5:]:
        prompt += f"<|user|>\n{user_msg}<|end|>\n<|assistant|>\n{bot_msg}<|end|>\n"
    prompt += f"<|user|>\n{message}<|end|>\n<|assistant|>\n"

    # 调用模型生成回复
    try:
        response = pipe.generate(prompt, max_new_tokens=256)
        # 清洗一下回复内容，避免重复打印提示词
        response = response.replace(prompt, "").strip()
        return response
    except Exception as e:
        return f"模型推理出错: {e}"

# 3. 启动Gradio界面
demo = gr.ChatInterface(
    fn=chat_with_model,
    title="🤖 本地OpenVINO聊天机器人",
    description="这是一个运行在本地CPU上的OpenVINO聊天机器人。",
    theme="soft",
)

if __name__ == "__main__":
    demo.launch()