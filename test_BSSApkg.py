import pytest
from BSSApkg import Analysis


def test_api_token_name():
    with pytest.raises(ValueError):
        Analysis.load_data()