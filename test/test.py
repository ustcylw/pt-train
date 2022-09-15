import sys
import time
import os
import re



# ###################################################################
# 好用

# from prettytable import PrettyTable


# data = [
#     ["小明","01","男"],
#     ["小红","02","女"],
#     ["小黄","03","男"]
# ]

# for i in range(10):

#     table=PrettyTable(["姓名","学号","性别"])
#     table.add_row(data[i%3])
#     table.add_row(data[(i+1)%3])

#     os.system('clear')
#     sys.stdout.write("\r{0}".format(table.get_string()))
#     sys.stdout.flush() 
#     time.sleep(1)
# ###################################################################
# ###################################################################


# from PyUtils.viz.show_table import ShowTable


# data = [
#     ['1', 'server01', '服务器01', '172.16.0.1'],
#     ['2', 'server02', '服务器02', '172.16.0.2'],
#     ['3', 'server03', '服务器03', '172.16.0.3'],
#     ['4', 'server04', '服务器04', '172.16.0.4'],
#     ['5', 'server05', '服务器05', '172.16.0.5'],
#     ['6', 'server06', '服务器06', '172.16.0.6'],
#     ['7', 'server07', '服务器07', '172.16.0.7'],
#     ['8', 'server08', '服务器08', '172.16.0.8'],
#     ['9', 'server09', '服务器09', '172.16.0.9']
# ]

# table = ShowTable(
#     item_list=['编号', '云编号', '名称', 'IP地址'],
#     title='hello world!'
# )
# table.set_algin('0')
# table.set_border(border=True)
# table.set_junction_char(junction_char='+')
# table.set_horizontal_char(horizontal_char='-')
# table.set_vertical_char(vertical_char='|')

# # for i in range(20):
# #     table.add_row(data[i%9])
# #     table.fresh(clear=False)
# #     time.sleep(0.5)

# # for i in range(20):
# #     table.fresh(data[i%9], clear=True)
# #     time.sleep(0.5)


# table.show()
# # table.show_slice(start=2, end=5)


import logging
from tqdm import trange
from tqdm import tqdm, tqdm_notebook
from tqdm.contrib.logging import logging_redirect_tqdm, tqdm_logging_redirect
import time
from prettytable import PrettyTable
# from PyUtils.logs.logger import Logger
from PyUtils.logs.logger_v1 import Logger

def main():

    LOG = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # LOG = Logger(
    #     name='logger-consol-file',
    #     level=logging.INFO,
    #     log_surfix='logger-consol-file',
    #     log_dir='/data/ylw/code/pl_yolo_v5/test',
    #     log_stream=0
    # ).get_logger()
    # LOG = Logger(
    #     log_surfix='hello',
    #     log_dir='/data/ylw/code/pl_yolo_v5/test', 
    #     level='info', 
    #     log_stream=2
    # )
    LOG.info(f'hello logger.')
    
    data = [
        ["小明","01","男"],
        ["小红","02","女"],
        ["小黄","03","男"]
    ]


    with tqdm_logging_redirect():
    # with logging_redirect_tqdm():
        # for i in trange(10):
        for i, datai in enumerate(tqdm(data)):
        # for i, datai in enumerate(data):

            # if i%2 == 0:
            table=PrettyTable(["姓名","学号","性别"])
            table.add_row(data[i%3])
            table.add_row(data[(i+1)%3])

            # os.system('clear')
            # sys.stdout.write("\r{0}".format(table.get_string()))
            # sys.stdout.flush() 
            LOG.info(f"\n{table.get_string()}")
            time.sleep(0.5)


if __name__ == '__main__':
    main()
    
    # logging restored
