import gradio as gr
import openvino_genai
from openvino_genai import LLMPipeline

css = '''
.gradio-container {
    max-width: 1850px !important;
    margin: 20px auto !important;
}
.message {
    padding: 10px !important;
    font-size: 14px !important;
}
/* 新增聊天区域滚动条 */
.chatbot {
    max-height: 70vh !important;
    overflow-y: auto !important;
}
/* Markdown 样式修复 */
.message h1, .message h2, .message h3, .message h4, .message h5, .message h6 {
    margin: 10px 0 !important;
    font-weight: bold !important;
    color: inherit !important;
}
.message h1 { font-size: 2em !important; }
.message h2 { font-size: 1.5em !important; }
.message h3 { font-size: 1.25em !important; }
.message h4 { font-size: 1.1em !important; }
.message code {
    background-color: #f0f0f0 !important;
    padding: 2px 6px !important;
    border-radius: 3px !important;
    font-family: monospace !important;
}
.message pre {
    background-color: #f5f5f5 !important;
    padding: 10px !important;
    border-radius: 5px !important;
    overflow-x: auto !important;
}
.message pre code {
    background-color: transparent !important;
    padding: 0 !important;
}
.message ul, .message ol {
    margin: 10px 0 !important;
    padding-left: 20px !important;
}
.message li {
    margin: 5px 0 !important;
}
.message p {
    margin: 8px 0 !important;
}
.message strong {
    font-weight: bold !important;
}
.message em {
    font-style: italic !important;
}
.message a {
    color: #007bff !important;
    text-decoration: underline !important;
}
.message blockquote {
    border-left: 4px solid #ddd !important;
    margin: 10px 0 !important;
    padding-left: 15px !important;
    color: #666 !important;
}
'''

# 加载模型（无需指定.xml文件）
pipe = LLMPipeline(
    models_path="E:\\models\\openvino\\DeepSeek-R1-Distill-Qwen-7B-int4-ov",
    device="GPU"
)

config = openvino_genai.GenerationConfig()
config.max_new_tokens = 2048        # 减少生成长度上限以避免超时
config.do_sample = True             
config.temperature = 0.5            # 降低温度值使输出更稳定
config.top_p = 1                    # 启用top-p采样并设置推荐值
config.repetition_penalty = 1.2     # 添加重复惩罚参数

# 启动管道
pipe.start_chat()

def streamer(subword: str):
    print(subword, end="", flush=True)
    # 添加终止符检测（如遇到<eos>则停止生成）
    if "</s>" in subword:
        return openvino_genai.StreamingStatus.STOP
    return openvino_genai.StreamingStatus.RUNNING

def predict(message, history):
    # Gradio ChatInterface 使用tuple格式: [(user_msg, assistant_response), ...]
    model_history = []
    
    # 正确处理Gradio的历史记录格式，添加错误处理
    for msg_pair in history:
        try:
            # 检查是否是有效的tuple格式
            if isinstance(msg_pair, (list, tuple)) and len(msg_pair) == 2:
                human_msg, assistant_msg = msg_pair
                model_history.append({"role": "user", "content": human_msg})
                if assistant_msg:
                    model_history.append({"role": "assistant", "content": assistant_msg})
            else:
                # 如果是其他格式，直接添加到历史记录
                model_history.append({"role": "user", "content": str(msg_pair)})
        except (ValueError, TypeError) as e:
            print(f"处理历史记录时出错: {e}, 消息对: {msg_pair}")
            continue
    
    # 添加当前用户消息
    model_history.append({"role": "user", "content": message})
    
    # 构建更符合模型期望的对话格式
    full_context = ""
    for msg in model_history:
        if msg["role"] == "user":
            full_context += f"用户: {msg['content']}\n"
        else:
            full_context += f"助手: {msg['content']}\n"
    
    full_response = ""
    
    try:
        for chunk in pipe.generate(full_context, config, streamer):
            # 更健壮的终止符处理
            if "</s>" in chunk:
                chunk = chunk.replace("</s>", "")
                if not chunk.strip():
                    break
            
            # 只有当有实际内容时才更新响应
            if chunk.strip():
                full_response += chunk
                yield full_response
    except Exception as e:
        print(f"生成过程中出错: {e}")
        if full_response.strip():
            yield full_response

# 启用自定义CSS样式
demo = gr.ChatInterface(
    fn=predict,
    type="tuples",
    examples=["你好", "介绍一下你自己"],
    description="与本地openvino模型聊天",
    css=css,
    chatbot=gr.Chatbot(
        label="对话历史",
        show_copy_button=True,
        bubble_full_width=False,
        height=600,
        render_markdown=True,
        sanitize_html=False
    )
)
                        
if __name__ == "__main__":
    demo.launch()