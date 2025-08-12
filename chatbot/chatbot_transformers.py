import gradio as gr
from transformers import pipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载本地模型
model_path = "Qwen/Qwen1.5-0.5B-Chat"  # 使用公开可用的0.5B模型

print("正在加载模型，这可能需要几分钟时间...")
# 使用pipeline方式加载模型
pipe = pipeline(
    "text-generation",
    model=model_path,
    tokenizer=model_path,
    torch_dtype=torch.float32,  # Always use float32 for CPU
    device_map="auto",
    trust_remote_code=True
)
print("模型加载完成!")

tokenizer = pipe.tokenizer

# 设置生成配置
gen_config = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1,
}

def predict(message, history):
    # 构造对话历史
    conversation = []
    for entry in history:
        try:
            if isinstance(entry, tuple) and len(entry) == 2:
                # Gradio's default format: (user_message, bot_response)
                conversation.append({"role": "user", "content": entry[0]})
                conversation.append({"role": "assistant", "content": entry[1]})
            elif isinstance(entry, dict):
                # Alternative format: {"content": "...", "response": "..."}
                conversation.append({"role": "user", "content": entry.get("content", "")})
                conversation.append({"role": "assistant", "content": entry.get("response", "")})
            else:
                print(f"Unexpected history entry format: {type(entry)} - {entry}")
                continue
        except Exception as e:
            print(f"Error processing history entry: {e}")
            continue
            
    conversation.append({"role": "user", "content": message})
    
    # 应用对话模板
    text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 生成响应
    outputs = pipe(text, **gen_config)
    response = outputs[0]["generated_text"][len(text):]
    return response

# 创建Gradio界面
demo = gr.ChatInterface(
    fn=predict,
    title="本地大模型聊天机器人",
    description="使用Hugging Face transformers库加载的本地大模型",
    examples=["你好", "介绍一下你自己", "你有什么功能？"],
    type="messages"  # 使用新的messages格式
)

if __name__ == "__main__":
    demo.launch()
