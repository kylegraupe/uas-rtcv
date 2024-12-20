"""Use this file to run all tests before executing the application."""

import unittest
import os

if __name__ == "__main__":
    test_suite = unittest.defaultTestLoader.loadTestsFromName('tests.settings_unit_tests')
    unittest.TextTestRunner(verbosity=2).run(test_suite)
