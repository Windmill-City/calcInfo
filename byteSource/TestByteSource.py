import unittest
import numpy as np
import os
import byteSource
import calcInfo


class ByteSourceTestCase(unittest.TestCase):
    """
    byteSource.py测试用例
    """
    def assertErrorIsAllowed(self, expected, actual, allow_error, filename=None):
        """
        判断实际与理论相对误差是否在允许范围内
        :param filename: 将误差写入文件路径
        :param expected: 理论数据
        :param actual: 实际数据
        :param allow_error: 允许的最大相对误差
        :return: None
        """
        # 将理论与实际数组中为0的元素替换为np.spacing(0),避免出现nan
        np_expected = np.where(np.array(expected) == 0, np.spacing(0), np.array(expected))
        actual = np.where(np.array(actual) == 0, np.spacing(0), np.array(actual))
        # 计算相对误差数组
        relative_error = (actual - np_expected)/np_expected
        # 相对误差数组最大值
        max_error = np.max(np.abs(relative_error))
        # 判断相对误差数组中绝对最大值是否小于允许的最大相对误差
        self.assertTrue(max_error < allow_error)
        if filename:
            with open(filename, 'a+') as test_file:
                test_file.write('%g, %g\n' % (max_error, allow_error))
                test_file.close()

    def test_gen_msg_arr(self):
        """
        仿真256元DMS消息流的单元测试
        :return: None
        """
        def test_gen_msg_arr_flow(prob, msg_size, allow_prob_error, allow_entropy_error, filename=None):
            """
            测试仿真256元DMS工作流
            :param prob: 理论概率数组
            :param msg_size: 信息长度
            :param allow_prob_error: 允许的对概率的最大相对误差
            :param allow_entropy_error: 允许的对信息熵的最大相对误差
            :return: None
            """
            # 调用byteSource函数生成指定字节长度的信息流
            msg = byteSource.gen_msg_arr(byteSource.CDF(prob), byteSource.rand_arr(msg_size))
            # 计算信息流概率数组
            msg_prob = calcInfo.probability(msg)
            # 计算信息流信息熵
            msg_entropy = calcInfo.entropy(msg_prob)
            # 判断概率数组以及信息熵相对误差数组中绝对最大值是否小于允许的最大相对误差
            self.assertErrorIsAllowed(prob, msg_prob, allow_prob_error, filename + '_prob.error.csv')
            self.assertErrorIsAllowed(calcInfo.entropy(prob), msg_entropy, allow_entropy_error,
                                      filename + '_entropy.error.csv')

        # 测试1：等概率分布, 信息长度为102400bytes, 概率, 信息熵允许的最大相对误差分别为0.2, 0.1
        expected_prob, size = np.full(256, 1/256), 102400
        prob_error, entropy_error = 0.25, 0.01
        test_gen_msg_arr_flow(expected_prob, size, prob_error, entropy_error, 'test/test1.1_byte.uniform_102400')
        # 测试2：概率分布不变, 信息长度为1024000bytes, 概率, 信息熵允许的最大相对误差分别为0.15, 0.01
        size = 1024000
        prob_error, entropy_error = 0.1, 1e-4
        test_gen_msg_arr_flow(expected_prob, size, prob_error, entropy_error,
                              'test/test1.2_byte.uniform_1024000')
        # 测试3：概率分布不变, 信息长度为10240000bytes, 概率, 信息熵允许的最大相对误差分别为0.1, 0.001
        size = 10240000
        prob_error, entropy_error = 0.05, 1e-5
        test_gen_msg_arr_flow(expected_prob, size, prob_error, entropy_error,
                              'test/test1.3_byte.uniform_10240000')
        # 测试4：0,1概率各为0.5, 其余为0的概率分布, 信息长度为102400bytes, 概率及信息熵允许最大相对误差分别为0.01, 0.0001
        expected_prob, size = np.array([0.5 if n <= 1 else 0 for n in range(256)]), 102400
        prob_error, entropy_error = 0.01, 1e-4
        test_gen_msg_arr_flow(expected_prob, size, prob_error, entropy_error,
                              'test/test1.4_byte0and1_both_p=0.5_102400')
        # 测试4：0概率为1, 其余为0的概率分布, 信息长度为1024bytes, 概率及信息熵允许最大相对误差均为1e-64
        expected_prob, size = np.array([1 if n == 0 else 0 for n in range(256)]), 1024
        prob_error, entropy_error = 1e-64, 1e-64
        test_gen_msg_arr_flow(expected_prob, size, prob_error, entropy_error,
                              'test/test1.5_byte0_with_p=1_1024')

    def test_generate_bDMS_extended_source(self):
        """
        二元DMS扩展为8次信源的单元测试
        :return: None
        """
        def test_flow(p_true, path, prob, msg_size, allow_prob_error, allow_entropy_error, filename=None):
            """
            二元DMS扩展为8次信源并进行仿真的测试
            :param p_true: 二元DMS1的概率
            :param path: 二元DMS扩展为8次信源的概率分布文件存储路径
            :param prob: 扩展后的理论概率数组
            :param msg_size: 扩展后的信源将要产生的信息的长度
            :param allow_prob_error: 允许的对概率的最大相对误差
            :param allow_entropy_error: 允许的对信息熵的最大相对误差
            :return: None
            """
            # 生成二元DMS扩展为8次信源的概率分布文件
            byteSource.generate_bDMS_extended_source_prob_file(p_true, path)
            # 读取所生成的概率分布文件的概率数组
            symbol_prob = byteSource.read_as_probability_distribution(path)
            # 生成指定字节长度的字符信息流
            msg = byteSource.gen_msg_arr(byteSource.CDF(symbol_prob), byteSource.rand_arr(msg_size))
            # 计算信息流概率数组
            msg_prob = calcInfo.probability(msg)
            # 判断扩展后的总概率是否为1
            self.assertAlmostEqual(np.sum(symbol_prob), 1)
            # 判断扩展后的概率分布是否与理论概率分布一致
            self.assertAlmostEqual(symbol_prob.all(), prob.all())
            # 计算信息流信息熵
            msg_entropy = calcInfo.entropy(msg_prob)
            # 判断概率数组以及信息熵相对误差数组中绝对最大值是否小于允许的最大相对误差
            self.assertErrorIsAllowed(symbol_prob, msg_prob, allow_prob_error, filename + '.extended_prob.error.csv')
            self.assertErrorIsAllowed(calcInfo.entropy(prob), msg_entropy, allow_entropy_error,
                                      filename + '.extended_entropy.error.csv')
            # 计算原二元DMS信息的概率
            prob_bit = calc_prob_bit(msg)
            # 判断还原的二元概率数组是否与理论一致
            self.assertErrorIsAllowed(np.array([1 - p_true, p_true]), prob_bit, allow_prob_error,
                                      filename + '.restored_prob.error.csv')
            self.assertErrorIsAllowed(calcInfo.entropy(np.array([1 - p_true, p_true])), calcInfo.entropy(prob_bit),
                                      allow_prob_error, filename + '.restored_entropy.error.csv')

        def calc_prob_bit(msg):
            """
            通过扩展的256元信源产生的消息流计算原二元DMS信息的概率
            :param msg: 生成的256元消息流
            :return: 二元DMS概率分布数组：numpy.array
            """
            # 统计总256个各个字符出现的总数
            (hist, bin_edges) = np.histogram(msg, bins=range(257))
            # 位总数为字符总数8倍
            num_bits = msg.size * 8
            # 计算位1, 0出现的数量
            num_1 = sum(hist[i]*bin(i).count('1') for i in range(256))
            num_0 = num_bits - num_1
            # 计算位1, 0的概率
            prob_0, prob_1 = num_0 / num_bits, num_1 / num_bits
            return np.array([prob_0, prob_1])

        # 测试1：位1的概率为0.5, 扩展后应为256字符等概率分布, 信息长度为102400bytes, 概率及信息熵允许最大相对误差分别为0.2, 0.1
        test_p_true, test_path = 0.5, 'test.dat'
        expected_prob, size = np.full(256, 1/256), 102400
        prob_error, entropy_error = 0.25, 0.01
        test_flow(test_p_true, test_path, expected_prob, size, prob_error, entropy_error,
                  'test/test2.1_bit1_both_p=0.5_102400')

        # 测试2：位1的概率为0, 扩展后应为字符0概率为1, 其余为0, 信息长度为1024bytes, 概率及信息熵允许最大相对误差均为1e-64
        test_p_true = 0
        expected_prob, size = np.array([1 if n == 0 else 0 for n in range(256)]), 1024
        prob_error, entropy_error = 1e-64, 1e-64
        test_flow(test_p_true, test_path, expected_prob, size, prob_error, entropy_error,
                  'test/test2.2_bit1_with_p=0_102400')

        # 测试3：位1的概率为1, 扩展后应为字符255概率为1, 其余为0, 信息长度为1024bytes, 概率及信息熵允许最大相对误差均为1e-64
        test_p_true = 1
        expected_prob, size = np.array([1 if n == 255 else 0 for n in range(256)]), 1024
        prob_error, entropy_error = 1e-64, 1e-64
        test_flow(test_p_true, test_path, expected_prob, size, prob_error, entropy_error,
                  'test/test2.3_bit1_with_p=1_102400')

        # 测试完毕移除测试文件
        os.remove(test_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
