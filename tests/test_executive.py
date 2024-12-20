"""
Use this file to run all tests before executing the application.
"""

import unittest
import os

if __name__ == "__main__":
    settings_unit_tests = unittest.defaultTestLoader.loadTestsFromName('tests.settings_unit_tests')

    unittest.TextTestRunner(verbosity=2).run(settings_unit_tests)

    # Not working yet
    # rtmp_integration_tests = unittest.defaultTestLoader.loadTestsFromName('tests.rtmp_stream_integration_tests')
    # unittest.TextTestRunner(verbosity=2).run(rtmp_integration_tests)
