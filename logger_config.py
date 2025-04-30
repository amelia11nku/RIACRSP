import os
import sys
import logging
import datetime

# 设置标准输出编码
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 创建日志目录
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

# 设置Matplotlib的日志级别为INFO，避免显示字体匹配的DEBUG信息
logging.getLogger('matplotlib').setLevel(logging.INFO)
# 设置PIL的日志级别为INFO，避免显示图像处理的DEBUG信息
logging.getLogger('PIL').setLevel(logging.INFO)


# 配置控制台日志处理器（输出INFO及以上级别的日志）
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.INFO)  
console_formatter = logging.Formatter('%(message)s')  
console_handler.setFormatter(console_formatter)

# 存储已创建的日志记录器
_loggers = {}

def setup_file_logger(module_name):
    """为指定模块设置文件日志处理器
    
    Args:
        module_name: 模块名称，用于日志文件名前缀
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 如果该模块的日志记录器已经存在，则直接返回
    if module_name in _loggers:
        return _loggers[module_name]
    
    # 创建模块特定的日志记录器
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    
    # 清除已有的处理器
    if logger.handlers:
        for handler in logger.handlers[:]: 
            logger.removeHandler(handler)
    
    # 添加控制台处理器
    logger.addHandler(console_handler)
    
    # 创建日志文件名（使用当前时间）
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'{module_name}_log_{current_time}.log')
    
    # 配置文件日志处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 阻止日志传递到父记录器，避免重复输出
    logger.propagate = False
    
    # 保存日志记录器以便复用
    _loggers[module_name] = logger
    
    return logger

# 为了兼容性，提供 get_logger 作为 setup_file_logger 的别名
get_logger = setup_file_logger
