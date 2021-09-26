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
        self.assertNpArrAlmostEqual(np.full(256, 0.00390625),
                                    calcInfo.probability(np.arange(256, dtype=np.uint8)))

    def test_self_info(self):
        self.assertNpArrAlmostEqual([0, 3.32192809, 2.32192809, 0.51457317],
                                    calcInfo.self_info(np.array([0, 0.1, 0.2, 0.7]), np.zeros(4)))

    def test_entropy(self):
        self.assertAlmostEqual(1.1567796494470395, calcInfo.entropy(np.array([0, 0.1, 0.2, 0.7])))

    def test_open_file_as_binary_arr(self):
        test_data = np.arange(256, dtype=np.uint8)

        with open('TestFile', 'wb+') as test_file:
            test_file.write(test_data.tobytes())
        test_file.close()

        self.assertNpArrAlmostEqual(test_data,
                                    calcInfo.open_file_as_binary_array(test_file.name))
        os.remove(test_file.name)

    def test_append_to_csv_by_row(self):
        data = [['path/to/input.file', '258', '6.38'],
                ['path/to/input.file', '258', '3.46']]

        for line in data:
            calcInfo.append_to_csv_by_row('result.csv', line)

        import csv
        with open('result.csv', 'r') as csv_file:
            for i, line in enumerate(csv.reader(csv_file)):
                self.assertEqual(data[i], line)
        csv_file.close()
        os.remove(csv_file.name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
