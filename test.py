# import logging
# import os
# import time
#
#
# def init_logging(log_filename):
#     # 确保日志目录存在
#     log_dir = os.path.dirname(log_filename)
#     if log_dir and not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#
#     # 配置日志基本设置
#     logging.basicConfig(
#         level=logging.DEBUG,  # 设置日志级别为DEBUG（最低级别）
#         format='%(asctime)s-%(levelname)s-%(message)s',  # 日志格式
#         datefmt='%y-%m-%d %H:%M',  # 日期格式
#         filename=log_filename,  # 日志文件名
#         filemode='w'  # 文件模式：'w'为覆盖写，'a'为追加写
#     )
#
#     # 添加控制台输出（可选）
#     console = logging.StreamHandler()
#     console.setLevel(logging.INFO)  # 控制台只显示INFO及以上级别的日志
#     formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
#     console.setFormatter(formatter)
#     logging.getLogger('').addHandler(console)
#
#     logging.info(f"日志系统初始化完成，日志文件: {log_filename}")
#
#
# def example_function():
#     """
#     示例函数，演示不同级别的日志记录
#     """
#     logging.debug("这是一个调试信息 - 通常用于开发阶段")
#     logging.info("这是一个普通信息 - 程序正常运行状态")
#     logging.warning("这是一个警告信息 - 可能有问题但程序仍可运行")
#
#     try:
#         # 模拟一个可能出错的操作
#         result = 10 / 2
#         logging.info(f"计算结果: {result}")
#
#         # 模拟一个错误
#         raise ValueError("这是一个模拟的错误")
#     except Exception as e:
#         logging.error(f"发生错误: {e}", exc_info=True)
#
#     logging.critical("这是一个严重错误信息 - 程序可能无法继续运行")
#
#
# def main():
#     """
#     主函数
#     """
#     # 初始化日志系统
#     log_file = "logs/app.log"  # 日志文件路径
#     init_logging(log_file)
#
#     # 记录程序开始
#     logging.info("程序开始运行")
#
#     # 执行示例函数
#     example_function()
#
#     # 模拟一些处理过程
#     for i in range(3):
#         logging.info(f"处理第 {i + 1} 项任务")
#         time.sleep(0.5)
#
#     # 记录程序结束
#     logging.info("程序运行结束")
#
#
# if __name__ == "__main__":
#     main()
# import logging
# import datetime
# import os
#
#
# def initLogging(logFilename):
#     """Init for logging
#     """
#     logging.basicConfig(
#                     level    = logging.DEBUG,
#                     format='%(asctime)s-%(levelname)s-%(message)s',
#                     datefmt  = '%y-%m-%d %H:%M',
#                     filename = logFilename,
#                     filemode = 'w');
#     console = logging.StreamHandler()
#     console.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
#     console.setFormatter(formatter)
#     logging.getLogger('').addHandler(console)
#
# path = 'log/'
# os.makedirs(path, exist_ok=True)   # 确保 log 目录存在
# current_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
# print(current_time)
# logname = os.path.join(path, current_time+'.txt')
# print(logname)
# initLogging(logname)

# from torch.distributions.laplace import Laplace
#
# a=Laplace(0,0.1)
#
# samples = a.sample((10,))
# print("10个随机样本:")
# print(samples)
# print(a)

a={}
a[1]={}
a[1][1]=1
a[1][2]=2
a[1]=0
print(a)
