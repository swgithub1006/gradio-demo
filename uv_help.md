# UV 包管理器常用命令

## 项目初始化

```bash
uv init                      # 初始化项目，生成 pyproject.toml
uv init --name myproject     # 指定项目名称
```

## 虚拟环境管理

```bash
uv venv                      # 创建虚拟环境（默认在 .venv 目录）
uv venv path/to/venv         # 指定虚拟环境路径
```

## 依赖管理

### 添加依赖

```bash
uv add pandas                # 添加单个依赖
uv add pandas numpy          # 添加多个依赖
uv add -r requirements.txt   # 从 requirements.txt 批量导入依赖到 pyproject.toml
uv add --dev pytest          # 添加开发依赖
```

### 移除依赖

```bash
uv remove requests           # 移除单个依赖
uv remove requests flask     # 移除多个依赖
```

### 安装/同步依赖

```bash
uv sync                      # 根据 pyproject.toml 和 uv.lock 安装依赖
uv sync --frozen             # 严格使用 lock 文件，不更新
uv pip install -r requirements.txt  # 从 requirements.txt 安装（不更新 pyproject.toml）
```

## 锁定文件管理

```bash
uv lock                      # 生成或更新 uv.lock 锁定文件
uv lock --upgrade            # 升级所有依赖到最新版本
uv lock --upgrade-package pandas  # 仅升级指定包
```

## 查看依赖

```bash
uv pip list                  # 列出已安装的包
uv pip freeze > requirements.txt  # 导出已安装的包到 requirements.txt
uv tree                      # 显示依赖树
```

## 运行脚本

```bash
uv run python main.py        # 在虚拟环境中运行脚本
uv run pytest                # 运行测试
```

## Python 版本管理

```bash
uv python install 3.12       # 安装指定 Python 版本
uv python list               # 列出已安装的 Python 版本
uv python pin 3.12           # 锁定项目 Python 版本
```

## 缓存管理

```bash
uv cache clean               # 清理缓存
```
