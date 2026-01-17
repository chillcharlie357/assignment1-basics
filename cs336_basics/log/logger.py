import logging
import os
from rich.logging import RichHandler
from cs336_basics.config import config

def _get_logger(name=None):
    if name is None:
        name = config.logging.name

    # 1. 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 允许记录所有级别的日志

    # 防止日志重复打印（如果该 logger 已被配置过）
    if logger.handlers:
        return logger

    # 2. 配置 RichHandler (用于控制台漂亮输出)
    console_handler = RichHandler(
        level=config.logging.level,
        show_path=config.logging.rich.show_path,
        rich_tracebacks=config.logging.rich.rich_tracebacks,
        markup=config.logging.rich.markup
    )

    # 3. 配置 FileHandler (用于记录到文件，方便以后查错)
    log_dir = config.logging.save_dir
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, config.logging.filename), encoding="utf-8")
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