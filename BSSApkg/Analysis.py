from typing import Any, Optional
import matplotlib
import matplotlib.pyplot as plt
import yaml
import requests
import pandas as pd
import numpy as np
import datetime

# Setting this option will print all collumns of a dataframe
pd.set_option('display.max_columns', None)
# Setting this option will print all of the data in a feature
pd.set_option('display.max_colwidth', None)


class Analysis():
    def __init__(self, analysis_config:str):
        CONFIG_PATHS = ['configs/system_config.yml', 'configs/user_config.yml']

        # add the analysis config to the list of paths to load
        paths = CONFIG_PATHS + [analysis_config]

        # initialize empty dictionary to hold the configuration
        config = {}

        # load each config file and update the config dictionary
        for path in paths:
            with open(path, 'r') as f:
                this_config = yaml.safe_load(f)
            config.update(this_config)

        self.config = config

        #initialize class attributes 
        self.BoosterVersion = []
        self.PayloadMass = []
        self.Orbit = []
        self.LaunchSite = []
        self.Outcome = []
        self.Flights = []
        self.GridFins = []
        self.Reused = []
        self.Legs = []
        self.LandingPad = []
        self.Block = []
        self.ReusedCount = []
        self.Serial = []
        self.Longitude = []
        self.Latitude = []
        



    def getBoosterVersion(self, data):
        """This function takes the dataset and uses the rocket column to call the API and append the data to the list

        Args:
            data ([type]): [description]
        """
        for x in data['rocket']:
            if x:
                try:
                    response = requests.get(f"https://api.spacexdata.com/v4/rockets/{x}")
                    if response.status_code == 200:
                        response_data = response.json()
                        self.BoosterVersion.append(response_data.get('name',None)
                    else:
                        print(f"Failed to get data for rocket {x}")
                        self.BoosterVersion.append(None)  # Or handle it in a way that makes sense for your application
                except requests.RequestException as e:
                    print(f"Request failed: {e}")
                    self.BoosterVersion.append(None)  # Or handle the error appropriately



    def getLaunchSite(self, data):
        """This function takes the dataset and uses the launchpad column to call the API and append the data to the list

        Args:
            data ([type]): [description]
        """
        for x in data['launchpad']:
            if x:
                try:
                    response = requests.get(f"https://api.spacexdata.com/v4/launchpads/{x}")
                    if response.status_code == 200:
                        response_data = response.json()
                        self.Longitude.append(response_data.get('longitude', None))
                        self.Latitude.append(response_data.get('latitude', None))
                        self.LaunchSite.append(response_data.get('name',None))
                    else:
                        print(f"Failed to get data for launchpad {x}")
                        self.Longitude.append(None)  # Handle missing data
                        self.Latitude.append(None)
                        self.LaunchSite.append(None)
                except requests.RequestException as e:
                    print(f"Request failed: {e}")
                    self.Longitude.append(None)  # Handle the error appropriately
                    self.Latitude.append(None)
                    self.LaunchSite.append(None)



    def getPayloadData(self, data):
        """  This function takes the dataset and uses the payloads column to call the API and append the data to the lists

        Args:
            data (_type_): _description_
        """
        for load in data['payloads']:
            if load:
                try:
                    response = requests.get(f"https://api.spacexdata.com/v4/payloads/{load}")
                    if response.status_code == 200:
                        response_data = response.json()
                        self.PayloadMass.append(response_data.get('mass_kg',None))
                        self.Orbit.append(response_data.get('orbit',None))
                    else:
                        print(f"Failed to get data for payloads {load}")
                        self.PayloadMass.append(None)
                        self.Orbit.append(None)
                except requests.RequestException as e:
                    print(f"Request failed: {e}")
                    self.PayloadMass.append(None)
                    self.Orbit.append(None)
                

    def getCoreData(self, data):
        """This function takes the dataset and uses the cores column to call the API and append the data such as 'Block','ReusedCount', 'Serial'  to the list

        Args:
            data (_type_): _description_
        """
        for core in data['cores']:
            if core['core'] is not None:
                try:
                    response = requests.get(f"https://api.spacexdata.com/v4/cores/{core['core']}")
                    if response.status_code == 200:
                        response_data = response.json()
                        self.Block.append(response_data.get('block', None))
                        self.ReusedCount.append(response_data.get('reuse_count', None))
                        self.Serial.append(response_data.get('serial', None))
                    else:
                        self.Block.append(None)
                        self.ReusedCount.append(None)
                        self.Serial.append(None)
                except requests.RequestException as e:
                    print(f"Request failed: {e}")
                    self.Block.append(None)
                    self.ReusedCount.append(None)
                    self.Serial.append(None)
            else:
                self.Block.append(None)
                self.ReusedCount.append(None)
                self.Serial.append(None)

            self.Outcome.append(str(core.get('landing_success', None)) + ' ' + str(core.get('landing_type', None)))
            self.Flights.append(core.get('flight', None))
            self.GridFins.append(core.get('gridfins', None))
            self.Reused.append(core.get('reused', None))
            self.Legs.append(core.get('legs', None))




    def load_data(self):
        """A function to fetch the initial dataset

        """
        
        static_json_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/API_call_spacex_api.json'
        response = requests.get(static_json_url)
        if response.status_code != 200:
            print("Failed to load data from API")
            return

        data = pd.json_normalize(response.json())

        # Preprocess the data
        data = data[['rocket', 'payloads', 'launchpad', 'cores', 'flight_number', 'date_utc']]
        data = data[data['cores'].map(len) == 1]
        data = data[data['payloads'].map(len) == 1]
        data['cores'] = data['cores'].map(lambda x: x[0])
        data['payloads'] = data['payloads'].map(lambda x: x[0])
        data['date'] = pd.to_datetime(data['date_utc']).dt.date
        data = data[data['date'] <= datetime.date(2020, 11, 13)]

        # Call data extraction methods
        self.getBoosterVersion(data)
        self.getLaunchSite(data)
        self.getPayloadData(data)
        self.getCoreData(data)

        # Compile the final data into a structured format
        self.launch_data = {
            'FlightNumber': list(data['flight_number']),
            'Date': list(data['date']),
            'BoosterVersion': self.BoosterVersion,
            'PayloadMass': self.PayloadMass,
            'Orbit': self.Orbit,
            'LaunchSite': self.LaunchSite,
            'Outcome': self.Outcome,
            'Flights': self.Flights,
            'GridFins': self.GridFins,
            'Reused': self.Reused,
            'Legs': self.Legs,
            'LandingPad': self.LandingPad,
            'Block': self.Block,
            'ReusedCount': self.ReusedCount,
            'Serial': self.Serial,
            'Longitude': self.Longitude,
            'Latitude': self.Latitude
        }
        self.launch_data_df = pd.DataFrame.from_dict(self.launch_data)



 

    def compute_analysis(self,launch_data_df):
        
        data_falcon9 = data1[data1['BoosterVersion']!='Falcon 1']
        data_falcon9.loc[:,'FlightNumber'] = list(range(1, data_falcon9.shape[0]+1))

        # Calculate the mean value of PayloadMass column
        mean_PlMass = data_falcon9['PayloadMass'].mean()
        # Replace the np.nan values with its mean value
        data_falcon9['PayloadMass'].replace(np.nan,mean_PlMass,inplace=True)

    

    def compute_analysis(self):
        """A function to manage to filter data for Falcon 9, generating new features and fill NaN values

        Returns:
            [type]: [description]
        """
        # Filter data for Falcon 9 launches (excluding Falcon 1)
        data_falcon9 = self.launch_data_df[self.launch_data_df['BoosterVersion'] != 'Falcon 1']

        # Renumber the FlightNumber starting from 1
        data_falcon9 = data_falcon9.reset_index(drop=True)
        data_falcon9['FlightNumber'] = range(1, len(data_falcon9) + 1)

        # Calculate the mean of the PayloadMass where values are not NaN
        mean_PlMass = data_falcon9['PayloadMass'].mean()

        # Replace NaN values in PayloadMass with the mean value
        data_falcon9['PayloadMass'].fillna(mean_PlMass, inplace=True)

        # Assuming data_falcon9 is your DataFrame after processing
        # Calculate landing outcomes
        landing_outcomes = data_falcon9['Outcome'].value_counts()

        # Identify bad outcomes (adjust indices based on your data)
        bad_outcomes = set(landing_outcomes.keys()[[1, 3, 5, 6, 7]])

        # Classify each landing
        landing_class = [0 if outcome in bad_outcomes else 1 for outcome in data_falcon9['Outcome']]

        # Add the classification to the DataFrame
        data_falcon9['Class'] = landing_class

        
        # You can return the modified DataFrame or store it in an attribute
        return data_falcon9







    def plot_data(self, save_path:Optional[str] = None) -> plt.Figure:
        pass