import pytest
from BSSApkg.Analysis import Analysis
import pandas as pd
import matplotlib

# def test_api_token_name():
#     with pytest.raises(ValueError):
#         Analysis.load_data()



# Write your unit tests here

# Create an instance of Analysis for testing
@pytest.fixture
def analysis_instance():
    analysis = Analysis('./analysis_config.yml')
    analysis.load_data()
    return analysis

def test_getBoosterVersion(analysis_instance):
    # Test if getBoosterVersion method returns a list of booster versions
    assert isinstance(analysis_instance.BoosterVersion, list)

def test_getLaunchSite(analysis_instance):
    # Test if getLaunchSite method returns a list of launch sites
    assert isinstance(analysis_instance.LaunchSite, list)

def test_getPayloadData(analysis_instance):
    # Test if getPayloadData method returns lists of PayloadMass and Orbit
    assert isinstance(analysis_instance.PayloadMass, list)
    assert isinstance(analysis_instance.Orbit, list)

def test_getCoreData(analysis_instance):
    # Test if getCoreData method returns lists of Block, ReusedCount, Serial, and others
    assert isinstance(analysis_instance.Block, list)
    assert isinstance(analysis_instance.ReusedCount, list)
    assert isinstance(analysis_instance.Serial, list)

def test_compute_analysis(analysis_instance):
    # Test if compute_analysis method returns a DataFrame
    data_falcon9 = analysis_instance.compute_analysis()
    assert isinstance(data_falcon9, pd.DataFrame)
    assert "Class" in data_falcon9.columns

def test_plot_data(analysis_instance):
    # Test if plot_data method returns a matplotlib Figure
    data_falcon9 = analysis_instance.compute_analysis()
    figure = analysis_instance.plot_data(data_falcon9)
    assert isinstance(figure, matplotlib.figure.Figure)

if __name__ == "__main__":
    pytest.main()
