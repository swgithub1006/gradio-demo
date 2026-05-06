import os
import gradio as gr
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v3",
    api_key="unused"
)

def predict(message, history):
    # 构建消息列表，包含历史记录
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})
    
    try:
        # 使用非流式输出以确保 Gradio 5.x 渲染稳定
        response = client.chat.completions.create(
            model="qwen3:4b",
            messages=messages,
            max_tokens=4096,
            stream=False
        )
        
        content = response.choices[0].message.content
        
        # 移除 <think> 标签内容，防止其干扰界面显示
        import re
        clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        # 如果过滤后为空，则显示原内容
        return clean_content if clean_content else content
                
    except Exception as e:
        print(f"Error in predict: {e}")
        return f"Error: {e}"

demo = gr.ChatInterface(predict, type="messages")

if __name__ == "__main__":
    demo.launch()