"""
Use this file to run all tests before executing the application.
"""

import unittest

from src import custom_logging

def execute_all_unit_tests() -> None:
    custom_logging.log_event(f'Running all unit tests.')
    settings_unit_tests = unittest.defaultTestLoader.loadTestsFromName('tests.settings_unit_tests')
    unittest.TextTestRunner(verbosity=1).run(settings_unit_tests)

if __name__ == "__main__":
    execute_all_unit_tests()

    # Not working yet
    # rtmp_integration_tests = unittest.defaultTestLoader.loadTestsFromName('tests.rtmp_stream_integration_tests')
    # unittest.TextTestRunner(verbosity=2).run(rtmp_integration_tests)
