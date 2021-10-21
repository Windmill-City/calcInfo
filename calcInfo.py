import timeit

import numpy as np
import csv


def probability(arr):
    """
    通过统计符号出现的频率计算符号出现的概率, 将字节作为符号划分单位
    :param arr: 字节数组(np.array)
    :return: 符号概率数组(np.array)
    """
    # 由于一个字节(byte)能表示 0-256 个符号, 所以区间为 (0, 257)
    # 返回在 0-256 区间内, 每个符号的出现的概率的数组
    return np.histogram(arr, bins=range(0, 257), density=True)[0]


def self_info(p_arr, out=None):
    """
    通过符号概率数组计算自信息量
    :param out: 存放结果的数组
    :param p_arr: 符号概率数组(np.array)
    :return: 自信息量数组(np.array)
    """
    # 通过概率计算自信息量
    return -np.log2(p_arr, out=out, where=np.where(p_arr, True, False))


def entropy(p_arr):
    """
    通过符号概率数组计算信息熵
    :param p_arr: 符号概率数组(np.array)
    :return: 信息熵
    """
    # 计算信息熵
    return np.sum(p_arr * self_info(p_arr))


def open_file_as_binary_array(path):
    """
    将文件打开为字节流
    :param path: 文件路径
    :return: 字节数组(np.array)
    """
    # 以 只读-字节流 模式打开文件
    return np.fromfile(path, dtype=np.uint8)


def append_to_csv_by_row(path, row):
    """
    将指定行附加到csv文件末尾
    :param path: csv文件路径
    :param row: 要附加的行
    :return: None
    """
    # 以附加模式打开文件
    with open(path, 'a+', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # 写入一行
        csv_writer.writerow(row)
    csv_file.close()


def parse_args():
    """
    根据命令行命令运行程序
    :return: None
    """
    import argparse
    # 程序简介
    parser = argparse.ArgumentParser(description='Calculate entropy of file')
    # Input 参数, 需要计算信息熵的文件路径
    parser.add_argument('INPUT', help='File to calc entropy', nargs='?')
    # Output 参数, 用于附加计算结果的CSV文件的路径
    parser.add_argument('OUTPUT', help='CSV file to append calc result', nargs='?')
    # verbose 参数, 用于控制log等级
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='show debug message')
    # method 参数, 用于控制输出概率数组和符合自信息数组
    parser.add_argument('-m', '--method', choices=['P', 'S'],
                        help='use method P(S) to print probability(self information)')
    # export-P 参数, 用于控制是否输出概率数组文件
    parser.add_argument('-p', '--export_P',
                        help='export csv file of probability to export_P')
    # export-S 参数, 用于控制是否输出自信息量数组文件
    parser.add_argument('-s', '--export_S',
                        help='export csv file of self information to export_S')
    # test 参数, 用于控制是否进行自动测试
    parser.add_argument('-t', '--test', action="store_true",
                        help='run auto test before calc')

    # 处理输入的命令
    args = parser.parse_args()

    import logging

    # 判断用户是否输入INPUT与OUTPUT
    if not args.INPUT or not args.OUTPUT:
        # 若输入操作(-v, -m, -p, -s),提示错误并返回
        if args.verbose or (not not(args.method or args.export_S or args.export_P)):
            print("Error: Option(-v, -m, -p, -s) required arguments: 'INPUT', 'OUTPUT'")
            return
    else:
        # 配置日志输出等级
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
        # 将文件读入到一个np.array中
        file_arr = open_file_as_binary_array(args.INPUT)
        # 计算概率数组
        p_arr = probability(file_arr)
        # 计算文件自信息量
        self_information = self_info(p_arr)
        # 计算文件信息熵
        file_entropy = entropy(p_arr)
        # 附加计算结果到CSV文件
        append_to_csv_by_row(args.OUTPUT, [args.INPUT, file_arr.size, file_entropy])

        # 启用详细信息输出文件名, 文件大小, 信息熵等信息
        globals()['file_arr'], globals()['p_arr'] = file_arr, p_arr
        calc_time = timeit.timeit("probability(file_arr)", "entropy(p_arr)", globals=globals(), number=1)
        logging.debug('Verbosity turned on\n' f'File:{args.INPUT}\nSize:{file_arr.size} bytes\n'
                      f'Data array:{file_arr}\nEntropy:{file_entropy} bit/byte\n'
                      f'Calc_time:{round(calc_time, 5)} sec')

        # 若需使用方法probability查看符合概率数组
        if args.method == 'P':
            # 输出概率数组
            logging.info(f'Probability:\n{p_arr}')
        # 若需使用方法self_info查看符合自信息数组
        elif args.method == 'S':
            # 输出自信息量
            logging.info(f'Self Info:\n{self_information}')

        # 若需将概率数组写入文件
        if not not args.export_P:
            with open(args.export_P, 'w', newline='') as P_csv:
                for i in range(p_arr.size):
                    P_csv.write('"%d","%g"\n' % (i, p_arr[i]))
                P_csv.close()
        # 若需将自信息量数组写入文件
        if not not args.export_S:
            with open(args.export_S, 'w', newline='') as self_info_csv:
                csv_writer = csv.writer(self_info_csv)
                csv_writer.writerows(np.squeeze(np.dstack((np.arange(256), self_information))))
                self_info_csv.close()

    # 判断是否进行单元测试
    if args.test:
        logging.info('Begin Unit Test')
        import subprocess
        subprocess.call(['python', 'TestCalcInfo.py'])


if __name__ == '__main__':
    parse_args()
