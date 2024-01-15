import pytest
from BSSApkg.Analysis import Analysis

# def test_api_token_name():
#     with pytest.raises(ValueError):
#         Analysis.load_data()



# Write your unit tests here

def test_load_data():
    # Test loading data from the API or a mock dataset
    analysis = Analysis("path_to_config.yml")
    analysis.load_data()
    # Add assertions to check if data is loaded correctly

def test_compute_analysis():
    # Test the compute_analysis method
    analysis = Analysis("path_to_config.yml")
    analysis.load_data()
    result = analysis.compute_analysis()
    # Add assertions to check if the result is as expected

def test_plot_data():
    # Test the plot_data method
    analysis = Analysis("path_to_config.yml")
    analysis.load_data()
    result = analysis.compute_analysis()
    figure = analysis.plot_data(result, save_path=None)
    # Add assertions to check if the figure is generated correctly
