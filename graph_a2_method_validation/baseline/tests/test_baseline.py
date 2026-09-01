import unittest

from graphcov.run.selection import METHODS


class BaselineTests(unittest.TestCase):
    def test_official_graph_a2_defaults(self):
        spec = METHODS["graph_a2"]
        self.assertEqual(spec.kwargs["k_neighbors"], 10)
        self.assertEqual(spec.kwargs["k_hops"], 2)


if __name__ == "__main__":
    unittest.main()

