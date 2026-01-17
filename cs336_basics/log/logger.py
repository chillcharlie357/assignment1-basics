import logging
import os
from rich.logging import RichHandler

def _get_logger(name="cs336"):
    # 1. 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 允许记录所有级别的日志

    # 防止日志重复打印（如果该 logger 已被配置过）
    if logger.handlers:
        return logger

    # 2. 配置 RichHandler (用于控制台漂亮输出)
    console_handler = RichHandler(
        level="INFO",               # 控制台只显示 INFO 及以上
        show_path=True,             # 显示是哪个文件在哪一行打印的
        rich_tracebacks=True,       # 极其美观的代码报错回溯
        markup=True                 # 允许在日志中使用 [bold red] 这种标签
    )

    # 3. 配置 FileHandler (用于记录到文件，方便以后查错)
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(f"{log_dir}/app.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG) # 文件里记录更详细的 DEBUG 信息
    
    # 纯文本格式，适合 grep 搜索
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 4. 将 Handler 添加到 logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# 创建一个全局单例供其他模块直接使用
logger = _get_logger()