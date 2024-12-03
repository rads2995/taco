import unittest

from taco.taco import taco

class TestTaco(unittest.TestCase):

    taquito: taco = taco()

    def test_taco(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
