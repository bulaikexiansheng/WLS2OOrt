import time
from contextlib import contextmanager

@contextmanager
def timer(message):
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{message}: {elapsed_time:.4f} seconds")
    yield elapsed_time


# 使用上下文定时器
# with timer("Some task"):
#     # 在这里执行你的任务
#     time.sleep(2)

# 输出：Some task: 2.0000 seconds
