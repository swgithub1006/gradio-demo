import ollama
import random

historys = [
    {"role": "system", "content": "你是一个AI助手。"},
    {"role": "assistant", "content": "你好。"}
]

def chat_to_ollama(history_contents): # 与ollama聊天
    response = ollama.Client(
        host="http://127.0.0.1:11434"
    ).chat(
        model='qwen3:4b', # 模型代号
        stream=False, # 是否使用流传输
        messages=history_contents, # 上下文
        options={ # 其他选项
            "temperature": 0.5,
            "num_ctx": 2048,
            "top_k": 50,
            "top_p": 0.9,
            "repeat_penalty": 1.2,
            "seed": random.randint(0, 1000000000)})
    return response

if __name__ == '__main__':
    # 这里用户输入了一句话：你是谁？
    historys.append({"role": "user", "content": "你是谁？"})
    # 模型输出回应
    print(chat_to_ollama(historys))