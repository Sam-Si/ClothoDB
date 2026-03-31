"""
Pytest configuration for ClothoDB tests.
"""

import pytest
from hypothesis import settings

# Configure Hypothesis settings for faster tests
settings.register_profile("fast", max_examples=50)
settings.register_profile("thorough", max_examples=200)
settings.register_profile("ci", max_examples=100, deadline=None)

# Use fast profile by default, CI profile in CI environment
settings.load_profile("fast")


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "property_based: marks tests as property-based tests")
