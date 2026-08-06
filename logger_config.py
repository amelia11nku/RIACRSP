import os
import sys
import logging
import datetime

# 设置标准输出编码
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 创建默认日志目录
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

# 设置第三方库的日志级别
logging.getLogger('matplotlib').setLevel(logging.INFO)
logging.getLogger('PIL').setLevel(logging.INFO)

# 配置控制台日志处理器
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.INFO)  
console_formatter = logging.Formatter('%(message)s')  
console_handler.setFormatter(console_formatter)

def setup_file_logger(module_name, log_file_path=None):
    """为指定模块设置文件日志处理器
    
    Args:
        module_name: 模块名称
        log_file_path: 指定的日志文件完整路径，若为None则使用默认logs目录
    """
    # 借助 logging 自带的单例模式获取 logger
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    
    # 清除已有的处理器（非常关键：允许我们后续动态更改输出文件位置）
    if logger.handlers:
        for handler in logger.handlers[:]: 
            logger.removeHandler(handler)
    
    # 添加控制台处理器
    logger.addHandler(console_handler)
    
    # 确定日志文件名和存放位置
    if log_file_path:
        log_file = log_file_path
        # 确保你指定的文件夹一定存在
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    else:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'{module_name}_log_{current_time}.log')
    
    # 配置文件日志处理器
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
    except OSError:
        logger.propagate = False
        return logger
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(file_handler)
    
    # 阻止日志传递到父记录器，避免重复输出
    logger.propagate = False
    
    return logger

get_logger = setup_file_logger
