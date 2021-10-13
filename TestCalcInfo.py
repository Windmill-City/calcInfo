import os
import unittest

import numpy as np

import calcInfo


class CalcInfoTestCase(unittest.TestCase):
    def assertNpArrAlmostEqual(self, expected, actual):
        self.assertAlmostEqual(np.array(expected).all(), np.array(actual).all())

    def test_probability(self):
        (expected_p, actual_p) = (np.full(8, 0.00390625), calcInfo.probability(np.arange(256, dtype=np.uint8)))
        self.assertNpArrAlmostEqual(expected_p, actual_p)
        (expected_p, actual_p) = (np.array([1 if n == 1 else 0 for n in range(256)]),
                                  calcInfo.probability(np.full(256, 1, dtype=np.uint8)))
        self.assertNpArrAlmostEqual(expected_p, actual_p)
        (expected_p, actual_p) = (np.array([0.5 if n <= 1 else 0 for n in range(256)]),
                                  calcInfo.probability(np.repeat(np.array([0, 1]).astype(np.uint8), 128)))
        self.assertNpArrAlmostEqual(expected_p, actual_p)

    def test_self_info(self):
        (expected_self_info, actual_self_info) = \
            (np.zeros(256), calcInfo.self_info(np.array([1 if n == 1 else 0 for n in range(256)]), np.zeros(256)))
        self.assertNpArrAlmostEqual(expected_self_info, actual_self_info)
        (expected_self_info, actual_self_info) = \
            (np.full(256, 8), calcInfo.self_info(np.full(256, 0.00390625), np.zeros(256)))
        self.assertNpArrAlmostEqual(expected_self_info, actual_self_info)
        (expected_self_info, actual_self_info) = \
            (np.array([1 if n <= 1 else 0 for n in range(256)]),
             calcInfo.self_info(np.array([0.5 if n <= 1 else 0 for n in range(256)]), np.zeros(256)))
        self.assertNpArrAlmostEqual(expected_self_info, actual_self_info)

    def test_entropy(self):
        expected_entropy, actual_entropy = 0, calcInfo.entropy(np.array([1 if n == 1 else 0 for n in range(256)]))
        self.assertNpArrAlmostEqual(expected_entropy, actual_entropy)
        expected_entropy, actual_entropy = 8, calcInfo.entropy(np.full(256, 0.00390625))
        self.assertNpArrAlmostEqual(expected_entropy, actual_entropy)
        expected_entropy, actual_entropy = 1, calcInfo.entropy(np.array([0.5 if n <= 1 else 0 for n in range(256)]))
        self.assertNpArrAlmostEqual(expected_entropy, actual_entropy)

    def test_open_file_as_binary_arr(self):

        def test_file(data):
            with open('TestFile', 'wb+') as file:
                file.write(data.tobytes())
            file.close()
            file_data = calcInfo.open_file_as_binary_array(file.name)
            os.remove(file.name)
            return file_data

        expected_data = np.arange(256, dtype=np.uint8)
        open_from_file_data = test_file(expected_data)
        self.assertNpArrAlmostEqual(expected_data, open_from_file_data)
        expected_data = np.repeat(np.array([0, 1]).astype(np.uint8), 128)
        open_from_file_data = test_file(expected_data)
        self.assertNpArrAlmostEqual(expected_data, open_from_file_data)
        expected_data = np.full(256, 1, dtype=np.uint8)
        open_from_file_data = test_file(expected_data)
        self.assertNpArrAlmostEqual(expected_data, open_from_file_data)

    def test_append_to_csv_by_row(self):

        def test_file(data):
            csv_file_data = []
            for line in data:
                calcInfo.append_to_csv_by_row('result.csv', line)

            import csv
            with open('result.csv', 'r') as csv_file:
                for i, line in enumerate(csv.reader(csv_file)):
                    csv_file_data.append(line)
                    self.assertAlmostEqual(data[i], line)
            csv_file.close()
            os.remove(csv_file.name)
            return csv_file_data

        expected_data = [['path/to/input.file', '258', '6.38'], ['path/to/input.file', '258', '3.46']]
        test_file(expected_data)
        expected_data = [['input/pic-city.png', '134', '2.2'], ['input/pic-city.png', '632', '3.3']]
        test_file(expected_data)
        expected_data = [['output/text-cn.7z', '189', '1.23'], ['input/text-cn.7z', '632', '6.3']]
        test_file(expected_data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
