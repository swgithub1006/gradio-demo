### 初始化项目
```bash
uv init  # 生成pyproject.toml
```

### 添加一个或多个依赖：
```bash
uv add pandas airtest
```

### 使用 remove 命令移除依赖包：
```bash
uv remove requests
```

### 使用 list 命令查看已安装的包：
```bash
uv pip list
```

### 虽然 UV 自身不直接提供生成 requirements.txt 的功能，但你可以使用 pip freeze 命令：
```bash
uv pip freeze > requirements.txt
```

### 用 UV 导入 requirements.txt 到pyproject.toml
```bash
uv add -r requirements.txt
```

### 从pyproject.toml自动安装依赖：
```bash
uv sync
```