"""
全双工语音对话系统 v3.2 - 日志模块
支持loguru回退到标准logging，日志输出到文件和控制台
"""
import logging
import sys
import os
from datetime import datetime

# 日志文件目录
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 日志文件名
LOG_FILE = os.path.join(LOG_DIR, f"voice_dialog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

try:
    from loguru import logger
    HAS_LOGURU = True

    # 移除默认处理器
    logger.remove()

    # 添加控制台输出（彩色）
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG",
        colorize=True
    )

    # 添加文件输出
    logger.add(
        LOG_FILE,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8"
    )

    logger.info(f"日志文件: {LOG_FILE}")

except ImportError:
    HAS_LOGURU = False

    # 创建标准logging的logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, encoding='utf-8')
        ]
    )
    logger = logging.getLogger("voice_dialog")
    logger.info(f"日志文件: {LOG_FILE}")

__all__ = ["logger", "HAS_LOGURU", "LOG_FILE"]