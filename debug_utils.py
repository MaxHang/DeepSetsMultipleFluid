"""
调试工具模块，用于控制 print 输出
"""
import tensorflow as tf

# 调试模式开关
DEBUG_MODE = False

# 每个 print 调用点的计数器
print_counters = {}

# 每个调用点最大输出次数
MAX_PRINTS = 100


def debug_print(*args, **kwargs):
    """
    调试打印函数，可以控制是否输出以及输出次数

    参数:
        *args: 要打印的参数
        **kwargs: 关键字参数
            caller_id: 调用点标识符，默认使用调用栈信息
            max_prints: 该调用点最大输出次数，默认为全局 MAX_PRINTS
    """
    if not DEBUG_MODE:
        return

    # 获取调用者信息
    import inspect
    caller_frame = inspect.currentframe().f_back
    caller_info = f"{caller_frame.f_code.co_filename}:{caller_frame.f_lineno}"

    # 获取调用点 ID
    caller_id = kwargs.pop('caller_id', caller_info)
    max_prints = kwargs.pop('max_prints', MAX_PRINTS)

    # 更新计数器
    if caller_id not in print_counters:
        print_counters[caller_id] = 0
    print_counters[caller_id] += 1

    # 检查是否超过最大打印次数
    if max_prints is None or print_counters[caller_id] <= max_prints:
        # 添加计数信息
        if max_prints is not None:
            tf.print(
                f"[DEBUG {print_counters[caller_id]}/{max_prints}]", *args, **kwargs)
        else:
            tf.print(f"[DEBUG {print_counters[caller_id]}]", *args, **kwargs)


def set_debug_mode(enabled=True):
    """设置调试模式开关"""
    global DEBUG_MODE
    DEBUG_MODE = enabled


def set_max_prints(max_count):
    """设置全局最大打印次数"""
    global MAX_PRINTS
    MAX_PRINTS = max_count


def reset_counters():
    """重置所有计数器"""
    global print_counters
    print_counters = {}
