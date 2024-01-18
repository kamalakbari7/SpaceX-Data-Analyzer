# Testing

## Test 1
The main test will be done automatically based on the defined action.yml file in GitHub. 

## Test 2 with pytest 
This test is based on the pytest and provided unit test inside test_Analysis.py file. Including tests for the following units:

1. analysis_instance()
2. getBoosterVersion(analysis_instance)
3. getLaunchSite(analysis_instance)
4. getPayloadData(analysis_instance)
5. getCoreData(analysis_instance)
6. compute_analysis(analysis_instance)
7. plot_data(analysis_instance)

to run this code you need to run this code in the terminal:
```bash
pytest test_Analysis.py
```