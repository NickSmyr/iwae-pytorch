import unittest


from utils.data_download import download_data
class MyTestCase(unittest.TestCase):
    def test_something(self):
        download_data("../../data")

    def dataset_stats(self):
        pass


if __name__ == '__main__':
    unittest.main()
