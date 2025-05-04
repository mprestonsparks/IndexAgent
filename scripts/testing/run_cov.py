import unittest
import coverage

def run_tests():
    # Start coverage
    cov = coverage.Coverage()
    cov.start()

    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover('tests')  # Assuming tests are in the 'tests' directory
    runner = unittest.TextTestRunner()
    runner.run(suite)

    # Stop coverage and save report
    cov.stop()
    cov.save()

    # Print coverage report
    coverage_percentage = cov.report()
    cov.html_report(directory='coverage_html_report')
    
    # Check if coverage is above 80%
    if coverage_percentage < 80:
        print("ERROR: Test coverage is below 80%!")
        exit(1)

if __name__ == '__main__':
    run_tests()