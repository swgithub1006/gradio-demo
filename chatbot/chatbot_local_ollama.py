import ollama
import gradio as gr

css = '''
.gradio-container {
    max-width: 1850px !important;
    margin: 20px auto !important;
}
.message {
    padding: 10px !important;
    font-size: 10px !important;
}
/* 新增聊天区域滚动条 */
.chatbot {
    max-height: 70vh !important;
    overflow-y: auto !important;
}
'''

def chat_to_ollama(history_contents):
    response = ollama.Client(
        host="http://127.0.0.1:11434"
    ).chat(
        model="gemma3:4b",
        # model="deepseek-r1:8b",
        stream=True,
        messages=history_contents,
        options={
            "temperature": 0.5,
            "num_ctx": 2048,
            "top_k": 30,
            "top_p": 0.8,
            "do_sample": True,
            "repeat_penalty": 1.2
        })
    return response

def chat_with_ollama(message, history):
    # Convert Gradio history format to Ollama format
    ollama_history = []
    
    # Handle the case where history is already in the correct format
    if history and isinstance(history[0], dict) and "role" in history[0]:
        ollama_history = history.copy()
    else:
        # Convert from tuple format if needed
        for human, assistant in history:
            ollama_history.append({"role": "user", "content": human})
            if assistant:
                ollama_history.append({"role": "assistant", "content": assistant})
    
    # Add current message
    ollama_history.append({"role": "user", "content": message})
    
    try:
        response = chat_to_ollama(ollama_history)
        assistant_response = ""
        
        for chunk in response:
            if chunk.get('message') and chunk['message'].get('content'):
                content = chunk['message']['content']
                assistant_response += content
                # print(f"Chunk received: {chunk}", end="", flush=True)  # Debug print
                print(content, end="", flush=True)  # Debug print
                yield assistant_response  # Yield only the content string
        
        # Ensure the final response is properly formatted for Gradio
        yield assistant_response
        
    except Exception as e:
        yield f"Error: {str(e)}"

# Initialize with system message
initial_history = [
    {"role": "system", "content": "你是一个AI助手。"},
    {"role": "assistant", "content": "你好，有什么我可以帮你的吗？"}
]

gr.ChatInterface(
    fn=chat_with_ollama,
    examples=["你好", "介绍一下你自己"],
    title="Ollama Chatbot",
    description="与本地Ollama模型聊天",
    type="messages",
    # css=css
).launch()
