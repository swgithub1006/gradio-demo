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
    font-size: 10px !important;
}
/* 新增聊天区域滚动条 */
.chatbot {
    max-height: 70vh !important;
    overflow-y: auto !important;
}
'''

# 加载模型（无需指定.xml文件）
pipe = LLMPipeline(
    models_path="D:\\Intel\\models\\DeepSeek-R1-Distill-Qwen-7B-int4-ov",
    device="GPU"
)

config = openvino_genai.GenerationConfig()
config.max_new_tokens = 2048        # 适当增加生成长度上限
config.do_sample = True             
config.temperature = 0.5            # 降低温度值使输出更稳定
config.top_p = 0.85                 # 启用top-p采样并设置推荐值
config.repetition_penalty = 1.1     # 添加重复惩罚参数

# 启动管道
pipe.start_chat()

def streamer(subword: str):
    print(subword, end="", flush=True)
    # 添加终止符检测（如遇到<eos>则停止生成）
    if "</s>" in subword:
        return openvino_genai.StreamingStatus.STOP
    return openvino_genai.StreamingStatus.RUNNING
    
def predict(message, history):
    full_response = ""
    for chunk in pipe.generate(message, config, streamer):
        # 过滤终止符并累积有效响应
        clean_chunk = chunk.replace("</s>", "").strip()
        if not clean_chunk:  # 忽略空内容
            continue
        full_response += clean_chunk
        yield full_response
    # 最终强制返回完整响应（防止提前终止）
    yield full_response

# 启用自定义CSS样式
demo = gr.ChatInterface(predict, type="messages",css=css) 
                        
if __name__ == "__main__":
    demo.launch()