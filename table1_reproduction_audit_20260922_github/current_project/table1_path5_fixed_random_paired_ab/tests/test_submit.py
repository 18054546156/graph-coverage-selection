from __future__ import annotations

import unittest

from submit import parse_job_id


class SubmitTests(unittest.TestCase):
    def test_parse_job_id(self) -> None:
        self.assertEqual(parse_job_id("Submitted batch job 12345"), "12345")

    def test_parse_job_id_rejects_noise(self) -> None:
        with self.assertRaises(ValueError):
            parse_job_id("submission failed")


if __name__ == "__main__":
    unittest.main()
