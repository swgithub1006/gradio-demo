import openvino_genai as ov_genai
import gradio as gr
import logging
from typing import Iterator

MODEL_PATH = "E:\\models\\openvino\\DeepSeek-R1-Distill-Qwen-7B-int4-ov"
DEVICE = "GPU"
MAX_HISTORY_LENGTH = 5
MAX_NEW_TOKENS = 4096

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info(f"正在加载模型: {MODEL_PATH}")
    pipe = ov_genai.LLMPipeline(MODEL_PATH, DEVICE)
    logger.info("模型加载成功")
except Exception as e:
    logger.error(f"模型加载失败: {e}")
    raise



def build_prompt(message: str, history: list) -> str:
    prompt = ""
    for item in history[-MAX_HISTORY_LENGTH * 2:]:
        if isinstance(item, dict):
            role = item.get("role", "user")
            content = item.get("content", "")
            if role == "user":
                prompt += f"<｜user｜>{content}<｜end＿send｜>"
            else:
                prompt += f"<｜assistant｜>{content}<｜end＿send｜>"
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            prompt += f"<｜user｜>{item[0]}<｜end＿send｜><｜assistant｜>{item[1]}<｜end＿send｜>"
    prompt += f"<｜user｜>{message}<｜end＿send｜><｜assistant｜>"
    return prompt

def stream_generate(message: str, history: list) -> Iterator[str]:
    prompt = build_prompt(message, history)
    logger.info(f"生成提示: {prompt[:200]}...")
    
    try:
        for token in pipe.generate(prompt, max_new_tokens=MAX_NEW_TOKENS):
            yield token
    except Exception as e:
        logger.error(f"生成失败: {e}")
        yield f"错误: {str(e)}"

def chat_with_model(message, history):
    full_response = ""
    for token in stream_generate(message, history):
        full_response += token
    
    if "<think>" in full_response:
        import re
        match = re.search(r"</think>\s*(.+)", full_response, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return full_response.strip()

demo = gr.ChatInterface(
    fn=chat_with_model,
    type="messages",
    title="🤖 本地OpenVINO聊天机器人",
    description="使用本地 DeepSeek-R1 模型提供动力",
)

if __name__ == "__main__":
    # 启动服务
    demo.launch()