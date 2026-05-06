import time
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
        # 使用非流式请求获取完整内容，以确保稳定性
        response = client.chat.completions.create(
            model="qwen3:4b",
            messages=messages,
            max_tokens=4096,
            stream=False
        )
        
        full_content = response.choices[0].message.content
        
        # 移除 <think> 标签内容
        import re
        clean_content = re.sub(r'<think>.*?</think>', '', full_content, flags=re.DOTALL).strip()
        if not clean_content:
            clean_content = full_content
        
        # 模拟流式输出（打字机效果）
        displayed_content = ""
        for char in clean_content:
            displayed_content += char
            yield displayed_content
            time.sleep(0.01) # 控制打字速度
                
    except Exception as e:
        print(f"Error in predict: {e}")
        yield f"Error: {e}"

demo = gr.ChatInterface(predict, type="messages")

if __name__ == "__main__":
    demo.launch()