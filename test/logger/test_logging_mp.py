from multiprocessing import Process

import logging_mp


def worker(name, level=None):
    # 如果没有设置 level，就使用全局默认等级
    logger = logging_mp.getLogger(name, level=level)

    logger.debug(f"[{name}] debug —— 调试细节")
    logger.info(f"[{name}] info —— 普通信息")
    logger.warning(f"[{name}] warning —— 警告但可以运行")
    logger.error(f"[{name}] error —— 出错但还能撑住")
    logger.critical(f"[{name}] critical —— 严重错误可能导致崩溃")

    logger.info(f"value A: {False}, value B: {123}")


if __name__ == "__main__":
    # 全局默认等级：INFO（即不显示 debug）
    logging_mp.basicConfig(level=logging_mp.INFO)

    main_logger = logging_mp.getLogger("main")
    main_logger.info("主进程启动")

    # 启动多个子进程：
    # 👉 worker-A：不设置等级，使用全局 INFO
    # 👉 worker-B：设置为 DEBUG，打印全部
    # 👉 worker-C：设置为 WARNING，只显示 warning 及以上
    # 👉 worker-D：不设置等级，也服从全局 INFO
    processes = [
        Process(target=worker, args=("worker-A",)),  # 使用全局等级 INFO
        Process(target=worker, args=("worker-B", logging_mp.DEBUG)),  # 单独设置为 DEBUG
        Process(
            target=worker, args=("worker-C", logging_mp.WARNING)
        ),  # 单独设置为 WARNING
        Process(target=worker, args=("worker-D",)),  # 使用全局等级 INFO
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    main_logger.info("主进程结束")
