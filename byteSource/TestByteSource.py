import unittest
import numpy as np
import os
import byteSource
import calcInfo


class ByteSourceTestCase(unittest.TestCase):
    def assertErrorIsAllowed(self, expected, actual, allow_error):
        expected = np.where(np.array(expected) == 0, np.spacing(1), np.array(expected))
        actual = np.where(np.array(actual) == 0, np.spacing(1), np.array(actual))
        relative_error = (actual - expected)/expected
        self.assertTrue(np.max(np.abs(relative_error)) < allow_error)

    def test_gen_msg_arr(self):

        def test_gen_msg_arr_flow(prob, entropy, msg_size, allow_prob_error, allow_entropy_error):
            msg = byteSource.gen_msg_arr(byteSource.CDF(prob), byteSource.rand_arr(msg_size))
            msg_prob = calcInfo.probability(msg)
            msg_entropy = calcInfo.entropy(msg_prob)
            self.assertErrorIsAllowed(prob, msg_prob, allow_prob_error)
            self.assertErrorIsAllowed(entropy, msg_entropy, allow_entropy_error)

        expected_prob, expected_entropy, size = np.full(256, 1/256), 8, 102400
        prob_error, entropy_error = 0.20, 0.1
        test_gen_msg_arr_flow(expected_prob, expected_entropy, size, prob_error, entropy_error)

        size = 1024000
        prob_error, entropy_error = 0.15, 0.01
        test_gen_msg_arr_flow(expected_prob, expected_entropy, size, prob_error, entropy_error)

        size = 10240000
        prob_error, entropy_error = 0.10, 0.001
        test_gen_msg_arr_flow(expected_prob, expected_entropy, size, prob_error, entropy_error)

        expected_prob, expected_entropy, size = np.array([0.5 if n <= 1 else 0 for n in range(256)]), 1, 102400
        prob_error, entropy_error = 0.01, 0.0001
        test_gen_msg_arr_flow(expected_prob, expected_entropy, size, prob_error, entropy_error)

        expected_prob, expected_entropy, size = np.array([1 if n == 0 else 0 for n in range(256)]), 0, 1024
        prob_error, entropy_error = 1e-64, 1e-64
        test_gen_msg_arr_flow(expected_prob, expected_entropy, size, prob_error, entropy_error)

    def test_generate_bDMS_extended_source_prob_file(self):

        def test_flow(p_true, path, prob, entropy, msg_size, allow_prob_error, allow_entropy_error):
            byteSource.generate_bDMS_extended_source_prob_file(p_true, path)
            symbol_prob = byteSource.read_as_probability_distribution(path)
            msg = byteSource.gen_msg_arr(byteSource.CDF(symbol_prob), byteSource.rand_arr(msg_size))
            compute_prob_of_01(msg)
            msg_prob = calcInfo.probability(msg)
            msg_entropy = calcInfo.entropy(msg_prob)
            self.assertErrorIsAllowed(symbol_prob, msg_prob, allow_prob_error)
            self.assertErrorIsAllowed(entropy, msg_entropy, allow_entropy_error)
            self.assertErrorIsAllowed(np.array([1 - p_true, p_true]), compute_prob_of_01(msg), allow_prob_error)
            self.assertAlmostEqual(symbol_prob.all(), prob.all())
            self.assertAlmostEqual(np.sum(symbol_prob), 1)

        def compute_prob_of_01(msg):
            (hist, bin_edges) = np.histogram(msg, bins=range(257))
            num_bits = msg.size * 8
            num_1 = np.sum(hist[i]*bin(i).count('1') for i in range(256))
            num_0 = num_bits - num_1
            prob_0, prob_1 = num_0 / num_bits, num_1 / num_bits
            return np.array([prob_0, prob_1])

        test_p_true, test_path = 0.5, 'test.dat'
        expected_prob, expected_entropy, size = np.full(256, 1/256), 8, 102400
        prob_error, entropy_error = 0.2, 0.1
        test_flow(test_p_true, test_path, expected_prob, expected_entropy, size, prob_error, entropy_error)

        test_p_true = 0
        expected_prob, expected_entropy, size = np.array([1 if n == 0 else 0 for n in range(256)]), 0, 1024
        prob_error, entropy_error = 1e-64, 1e-64
        test_flow(test_p_true, test_path, expected_prob, expected_entropy, size, prob_error, entropy_error)

        test_p_true = 1
        expected_prob, expected_entropy, size = np.array([1 if n == 255 else 0 for n in range(256)]), 0, 1024
        prob_error, entropy_error = 1e-64, 1e-64
        test_flow(test_p_true, test_path, expected_prob, expected_entropy, size, prob_error, entropy_error)

        os.remove(test_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
