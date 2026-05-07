# Gradio Demo

基于 Gradio 的 AI 聊天机器人演示项目，支持多种后端模型和 RAG 功能。

## 项目结构

```
gradio-demo/
├── chatbot/                  # 聊天机器人实现
│   ├── chatbot_transformers.py        # 基于 Transformers 模型
│   ├── chatbot_local_ollama.py        # 本地 Ollama 模型
│   ├── chatbot_local_ovms.py          # 本地 OpenVINO 模型
│   ├── chatbot_ov_ds_distill_qwen_7b.py  # OpenVINO + DeepSeek Distill Qwen 7B
│   ├── chatbot_ov_qwen3.py           # OpenVINO + Qwen3
│   ├── chatbot_deepseek.py           # DeepSeek API
│   └── chatbot_rag_text.py           # RAG 文本检索
├── pyproject.toml            # 项目配置和依赖
├── requirements.txt          # 依赖列表
├── uv.lock                   # 锁定文件
├── uv_help.md                # UV 包管理器使用指南
└── README.md                 # 本文档
```

## 环境要求

- Python >= 3.10

## 快速开始

### 使用 UV 包管理器（推荐）

```bash
# 创建虚拟环境
uv venv

# 安装依赖
uv sync

# 运行聊天机器人
uv run python chatbot/chatbot_local_ollama.py
```

### 使用 pip

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 运行聊天机器人
python chatbot/chatbot_local_ollama.py
```

## 可用的聊天机器人

| 文件 | 描述 |
|------|------|
| `chatbot_local_ollama.py` | 使用本地 Ollama 服务 |
| `chatbot_transformers.py` | 使用 HuggingFace Transformers 模型 |
| `chatbot_local_ovms.py` | 使用本地 OpenVINO Model Server |
| `chatbot_deepseek.py` | 使用 DeepSeek API |
| `chatbot_rag_text.py` | 基于文本检索的 RAG 实现 |
| `chatbot_ov_ds_distill_qwen_7b.py` | OpenVINO + DeepSeek Distill Qwen 7B |
| `chatbot_ov_qwen3.py` | OpenVINO + Qwen3 |

## UV 包管理器使用

参考 [uv_help.md](uv_help.md) 获取更多 UV 命令示例。
