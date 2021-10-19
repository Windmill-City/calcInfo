import numpy as np


# noinspection PyPep8Naming
def CDF(symbol_prob):
    """
    CDF - Cumulative probability Distribution Function
    计算给定概率分布P的累积概率分布F
    :param symbol_prob: 概率分布数组:numpy.array
    :return: 累积概率分布F:numpy.array
    """
    return symbol_prob.cumsum()


def rand_arr(size):
    """
    生成[0,1]区间均匀分布的随机实数f
    :param size: 随机实数数组大小
    :return: 随机实数数组f:numpy.array
    """
    return np.random.uniform(size=size)


def gen_msg_arr(symbol_cumsum, symbol_random):
    """
    根据指定累积概率分布数组和随机数数组生成信源数组
    :param symbol_cumsum: 累积概率分布数组:numpy.array
    :param symbol_random: 随机数数组:numpy.array
    :return: 信源数组:numpy.array
    """

    """
    Step 3: 生成符号以下条件的消息符号
   $$ 
   x = 
   \begin{cases}
      0, \quad f \leqslant F(0) \\
      i, \quad F(i-1) < f \leqslant F(i), \quad (0 < i \leqslant q)  
   \end{cases}
   $$
   
   即当
   symbol_cumsum[i-1] < f < symbol_cumsum[i] - (1)
   时
   输出i的值
   
   `np.searchsorted(a,v)`返回一个数组
   其中的值为 数组`v`中`元素(element)` 要 **顺序插入** 已排序数组`a` 中的`索引(index)`信息
   [ele_0 -> index_0,
    ele_1 -> index_1,
    ele_2 -> index_2,
    ...
    ele_n -> index_n]
    即要将`ele_0`顺序插入`a`中, 我们需要将`ele_0`插入到`a[index_0]`的后面
    所以这个`ele_0`自然符合模式 a[i-1] < v < a[i], 即符合模式 (1)
    将`a`置换为累积概率分布F(i), `ele_0`置换为随机实数f
    `np.searchsorted`的过程就是在累积概率分布F(i)中随机取出符合指定概率分布P的消息
    而`index_0`恰好对应的就是随机取出的消息
    """
    return np.searchsorted(symbol_cumsum, symbol_random)


def save_as_byte_source(path, byte_source):
    """
    将信源数组保存到指定路径
    :param path: 保存路径
    :param byte_source: 信源数组:numpy.array
    :return: None
    """
    # 以覆写模式打开文件
    with open(path, 'w+b') as byte_source_file:
        byte_source_file.write(byte_source)
        byte_source_file.flush()
    byte_source_file.close()


def read_as_probability_distribution(path):
    """
    从CSV文件中读入概率分布数组
    :param path: CSV文件路径
    :return: 概率分布数组:numpy.array
    """
    import csv
    # 概率分布数组(0-255)
    symbol_probability = np.zeros(256, dtype=float)
    # 以读模式打开文件
    with open(path, 'r') as csv_file:
        for line in csv.reader(csv_file):
            symbol, probability = line
            symbol_probability[int(symbol)] = probability
    csv_file.close()
    return symbol_probability


def save_as_csv(path, data):
    """
    将数据保存到指定路径的CSV文件当中
    :param path: CSV文件路径
    :param data: 要保存的数据
    :return: None
    """
    import csv
    # 以覆写模式打开文件
    with open(path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data)
        csv_file.flush()
    csv_file.close()


def parse_args():
    """
    处理命令行参数, 并根据参数执行程序
    :return: None
    """
    import argparse
    # 程序简介
    parser = argparse.ArgumentParser(description='Generate information source by probability distribution')
    # PROBABILITY_CSV - 概率分布数据, CSV格式
    parser.add_argument('PROBABILITY_CSV', help='Probability distribution, in CSV format', nargs='?')
    # BYTE_SOURCE - 信源数据保存路径
    parser.add_argument('BYTE_SOURCE', help='Where to save the byte source data', nargs='?')
    # SIZE - 产生的信源的数据个数
    parser.add_argument('SIZE', help='How much msg we need to generate', nargs='?', type=int)
    # -v 输出debug信息
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='log level set to [logging.DEBUG]')
    # -F <CDF_CSV> - 输出累积概率分布数组到指定文件, CSV格式
    parser.add_argument('-F', '--CDF_CSV', action='store', nargs=1,
                        help='Save CDF array to file, in CSV format')
    # -R <RAND_CSV> - 输出随机数数组到指定文件, CSV格式
    parser.add_argument('-R', '--RAND_CSV', action='store', nargs=1,
                        help='Save random array to file, in CSV format')
    # -t - 运行测试
    parser.add_argument('-t', '--test', action="store_true",
                        help='run tests')

    # 处理命令行参数
    args = parser.parse_args()

    import logging
    # 配置日志输出等级
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logging.debug(f'Command Line:{args}')

    if args.test:
        logging.info('Begin Unit Test')
        import subprocess
        subprocess.call(['python', 'TestByteSource.py'])
        return
    else:
        if not args.PROBABILITY_CSV or not args.BYTE_SOURCE or not args.SIZE:
            parser.error('Missing required argument(s)')

    # 从指定路径读入概率分布P
    logging.debug(f'Read Probability from:{args.PROBABILITY_CSV}')
    symbol_prob = read_as_probability_distribution(args.PROBABILITY_CSV)
    # Step 1: 计算累积概率分布F
    logging.debug('Evaluating CDF')
    symbol_cumsum = CDF(symbol_prob)
    # 保存累积概率分布到文件中
    if args.CDF_CSV:
        save_as_csv(args.CDF_CSV[0], symbol_cumsum.reshape([symbol_cumsum.size, 1]))
        logging.info(f'Saved CDF data to:{args.CDF_CSV[0]}')
    # Step 2: 生成size个在[0,1]之间均匀分布的随机实数f
    logging.debug('Generating Rand array')
    symbol_random = rand_arr(args.SIZE)
    # 保存随机数数组到文件中
    if args.RAND_CSV:
        save_as_csv(args.RAND_CSV[0], symbol_random.reshape([symbol_random.size, 1]))
        logging.info(f'Saved Rand data to:{args.RAND_CSV[0]}')
    # Step 3: 生成消息符号
    logging.debug('Generating byte source')
    byte_source = gen_msg_arr(symbol_cumsum, symbol_random)
    # 保存信源数据到文件中
    save_as_byte_source(args.BYTE_SOURCE, byte_source)
    logging.info(f'Saved ByteSource to:{args.BYTE_SOURCE}')


if __name__ == '__main__':
    parse_args()
