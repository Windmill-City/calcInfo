import os
import unittest

import numpy as np

import calcInfo


class CalcInfoTestCase(unittest.TestCase):
    def assertNpArrAlmostEqual(self, expected, actual):
        expected = np.array(expected)
        assert expected.size == actual.size, f'Expected:{expected} Actual:{actual}'
        for i in range(expected.size):
            self.assertAlmostEqual(expected[i], actual[i])

    def test_probability(self):
        expected_p, test_p = np.full(256, 0.00390625), calcInfo.probability(np.arange(256, dtype=np.uint8))
        self.assertNpArrAlmostEqual(expected_p, test_p)
        np.set_printoptions(threshold=5, edgeitems=3)
        print("\n expected_p:", expected_p, "\n test_p:", test_p)

    def test_self_info(self):
        expected_self_info, test_self_info = np.array([0, 3.32192809, 2.32192809, 0.51457317]), \
                                             calcInfo.self_info(np.array([0, 0.1, 0.2, 0.7]), np.zeros(4))
        self.assertNpArrAlmostEqual(expected_self_info, test_self_info)
        print("\n expected_self_info:", expected_self_info, "\n test_self_info:", test_self_info)

    def test_entropy(self):
        expected_entropy, test_entropy = 1.1567796494470395, calcInfo.entropy(np.array([0, 0.1, 0.2, 0.7]))
        self.assertAlmostEqual(expected_entropy, test_entropy)
        print("\n expected_entropy:", expected_entropy, "\n test_entropy:", test_entropy)

    def test_open_file_as_binary_arr(self):
        expected_data = np.arange(256, dtype=np.uint8)

        with open('TestFile', 'wb+') as test_file:
            test_file.write(expected_data.tobytes())
        test_file.close()
        open_from_file_data = calcInfo.open_file_as_binary_array(test_file.name)
        self.assertNpArrAlmostEqual(expected_data, open_from_file_data)
        os.remove(test_file.name)

        np.set_printoptions(threshold=5, edgeitems=8)
        print("\n expected_data:", expected_data, "\n open_from_file_data:", open_from_file_data)

    def test_append_to_csv_by_row(self):
        expected_data = [['path/to/input.file', '258', '6.38'], ['path/to/input.file', '258', '3.46']]
        append_to_csv_data = []

        for line in expected_data:
            calcInfo.append_to_csv_by_row('result.csv', line)

        import csv
        with open('result.csv', 'r') as csv_file:
            for i, line in enumerate(csv.reader(csv_file)):
                append_to_csv_data.append(line)
                self.assertEqual(expected_data[i], line)
        csv_file.close()
        os.remove(csv_file.name)

        print("\n expected_data:", expected_data, "\n append_to_csv_data:", append_to_csv_data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
