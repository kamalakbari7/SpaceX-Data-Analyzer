from typing import Any, Optional
import matplotlib
import matplotlib.pyplot as plt
import yaml
import requests
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import os
import logging
from pathlib import Path
import os


# Setting this option will print all collumns of a dataframe
pd.set_option('display.max_columns', None)
# Setting this option will print all of the data in a feature
pd.set_option('display.max_colwidth', None)

import logging

# Explicitly configure the root logger
logging.basicConfig(level=logging.INFO)

# Configure urllib3 logger to WARNING to suppress DEBUG messages
logging.getLogger('urllib3').setLevel(logging.WARNING)

logger = logging.getLogger()


class Analysis():
    def __init__(self, analysis_config:str):
        # CONFIG_PATHS = ['./configs/system_config.yml', './configs/user_config.yml']

        # # add the analysis config to the list of paths to load
        # paths = CONFIG_PATHS + [analysis_config]

        # # initialize empty dictionary to hold the configuration
        # config = {}

        # for path in paths:
        #     try:
        #         with open(path, 'r') as f:
        #             this_config = yaml.safe_load(f)
        #         config.update(this_config)
        #     except FileNotFoundError:
        #         logger.error(f"File not found: {path}")
        #         raise
        #     except yaml.YAMLError as e:
        #         logger.error(f"Error parsing YAML file: {path},{e}")
        #         raise

        # Get the directory of the current script
        dir_path = Path(__file__).parent.parent

        # System and user configuration paths
        system_config_path = dir_path / 'configs' / 'system_config.yml'
        user_config_path = dir_path / 'configs' / 'user_config.yml'

        # Add the analysis-specific configuration path
        analysis_config_path = dir_path / 'configs' / analysis_config

        # Initialize configuration dictionary
        self.config = {}

        # Load configurations from each file
        for path in [system_config_path, user_config_path, analysis_config_path]:
            try:
                with open(path, 'r') as file:
                    self.config.update(yaml.safe_load(file))
            except FileNotFoundError as e:
                logging.error(f"Configuration file not found: {path}")
                raise e
            except yaml.YAMLError as e:
                logging.error(f"Error parsing YAML file: {path}, {e}")
                raise e
        
        # self.config = config

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
        self.Block = []
        self.ReusedCount = []
        self.Serial = []
        self.Longitude = []
        self.Latitude = []


    def ensure_save_path(self, plot_key):    
        """
        Ensure the save path for a plot exists, and if not, create it. The method checks the configuration
        for a specified plot key ('plot1', 'plot2', etc.) and creates the directory if it doesn't exist.

        Parameters
        ----------
        plot_key : str
             The key for the plot configuration (e.g., 'plot1', 'plot2').

         Returns
         -------
         str
             The save path for the plot.
        """
        default_save_path = './plot/'  # Default path if not specified in the config
        save_path = self.config.get('plot', {}).get(plot_key, {}).get('save_path', default_save_path)

        # Create the directory if it does not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        
        return save_path    


    def getBoosterVersion(self, data):
        """
        This function takes the dataset and uses the rocket column to call the API and append the data to the list

        Parameters
        ----------
        data : pandas.DataFrame
             The dataset containing a 'rocket' column with rocket IDs.

        """
        for x in data['rocket']:
            if x:
                try:
                    response = requests.get(f"https://api.spacexdata.com/v4/rockets/{x}")
                    if response.status_code == 200:
                        logger.info("Request to SpaceX API was successful.")
                        response_data = response.json()
                        self.BoosterVersion.append(response_data.get('name',None))
                    else:
                        logger.warning(f"Request returned a non-200 status code. Failed to get data for rocket {x}")
                        self.BoosterVersion.append(None)  # Or handle it in a way that makes sense for your application
                except requests.RequestException as e:
                    logger.error(f"Request failed: {e}")
                    self.BoosterVersion.append(None)  # Or handle the error appropriately



    def getLaunchSite(self, data):
        """This function takes the dataset and uses the launchpad column to call the API and append the data to the list

        Parameters
        ----------
        data : pandas.DataFrame
            The dataset containing a 'launchpad' column with launchpad IDs.  

        """
       
        for x in data['launchpad']:
            if x:
                try:
                    response = requests.get(f"https://api.spacexdata.com/v4/launchpads/{x}")
                    if response.status_code == 200:
                        response_data = response.json()
                        logger.info("Request to SpaceX API was successful.")
                        self.Longitude.append(response_data.get('longitude', None))
                        self.Latitude.append(response_data.get('latitude', None))
                        self.LaunchSite.append(response_data.get('name',None))
                    else:
                        logger.warning(f"Request returned a non-200 status code. Failed to get data for launchpad {x}")
                        self.Longitude.append(None)  # Handle missing data
                        self.Latitude.append(None)
                        self.LaunchSite.append(None)
                except requests.RequestException as e:
                    logger.error(f"Request failed: {e}")
                    self.Longitude.append(None)  # Handle the error appropriately
                    self.Latitude.append(None)
                    self.LaunchSite.append(None)



    def getPayloadData(self, data):
        """This function takes the dataset and uses the payloads column to call the API and append the data to the lists

        Parameters
        ----------
         data : pandas.DataFrame
            The dataset containing a 'launchpad' column with launchpad IDs.

        """
        for load in data['payloads']:
            if load:
                try:
                    response = requests.get(f"https://api.spacexdata.com/v4/payloads/{load}")
                    if response.status_code == 200:
                        logger.info("Request to SpaceX API was successful.")
                        response_data = response.json()
                        self.PayloadMass.append(response_data.get('mass_kg',None))
                        self.Orbit.append(response_data.get('orbit',None))
                    else:
                        logger.warning(f"Request returned a non-200 status code. Failed to get data for PayLoad {load}")
                        self.PayloadMass.append(None)
                        self.Orbit.append(None)
                except requests.RequestException as e:
                    logger.error(f"Request failed: {e}")
                    self.PayloadMass.append(None)
                    self.Orbit.append(None)
                

    def getCoreData(self, data):
        """This function takes the dataset and uses the cores column to call the API and append the data such as 'Block','ReusedCount', 'Serial'  to the list

        Parameters
        ----------
        data : pandas.DataFrame
            The dataset containing a 'cores' column with core information.

        """
        for core in data['cores']:
            if core['core'] is not None:
                try:
                    response = requests.get(f"https://api.spacexdata.com/v4/cores/{core['core']}")
                    if response.status_code == 200:
                        logger.info("Request to SpaceX API was successful.")
                        response_data = response.json()
                        self.Block.append(response_data.get('block', None))
                        self.ReusedCount.append(response_data.get('reuse_count', None))
                        self.Serial.append(response_data.get('serial', None))
                    else:
                        logger.warning(f"Request returned a non-200 status code. Failed to get data for CoreData {core}")
                        self.Block.append(None)
                        self.ReusedCount.append(None)
                        self.Serial.append(None)
                except requests.RequestException as e:
                    logger.error(f"Request failed: {e}")
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


    def clear_data(self):
        """Clears the data lists to prepare for new data loading."""
        self.BoosterVersion = []
        self.PayloadMass = []
        self.Orbit = []
        self.LaunchSite = []
        self.Outcome = []
        self.Flights = []
        self.GridFins = []
        self.Reused = []
        self.Legs = []
        self.Block = []
        self.ReusedCount = []
        self.Serial = []
        self.Longitude = []
        self.Latitude = []

    def load_data(self):       
        """A function to fetch the initial dataset
        
        """
        self.clear_data()
        spacex_url="https://api.spacexdata.com/v4/launches/past"
        response = requests.get(spacex_url)
        if response.status_code != 200:
            logger.warning("Failed to load data from API")
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
            'Block': self.Block,
            'ReusedCount': self.ReusedCount,
            'Serial': self.Serial,
            'Longitude': self.Longitude,
            'Latitude': self.Latitude
        }
        # debugging statements
        for key, value in self.launch_data.items():
            print(f"Key: {key}, Length: {len(value)}")

        self.launch_data_df = pd.DataFrame.from_dict(self.launch_data)

    

    def compute_analysis(self):
        """A function to manage to filter data for Falcon 9, generating new features and fill NaN values

        Returns
        -------
        pandas.DataFrame
             A DataFrame containing filtered Falcon 9 launch data with new features and filled NaN values.    
        
       """
        # Checking if BoosterVersion column there is in the launch_data_df with assert
        assert 'BoosterVersion' in self.launch_data_df.columns, "BoosterVersion column is missing in the data"
        
        # Filter data for Falcon 9 launches (excluding Falcon 1)
        data_falcon9 = self.launch_data_df[self.launch_data_df['BoosterVersion'] != 'Falcon 1']

        # Checking if FlightNumber column there is in the launch_data_df with assert
        assert 'FlightNumber' in self.launch_data_df.columns, "FlightNumber column is missing in the data"
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



    def plot_data(self, data_falcon9) -> plt.Figure:
        """
        A function for visualizing the results of analyses. It includes two plots:
        1. A bar chart for success rate by orbit type.
        2. A line chart for success rate over years, using the DataFrame processed by compute_analysi

        Parameters
        ----------
        data_falcon9 : pd.DataFrame
            The DataFrame containing processed data for Falcon 9.

        Returns
        -------
        plt.Figure
            A matplotlib figure object containing the generated plots.     
   
        """
        plt.figure()
        # Load plot configurations from analysis_config.yml
        plot1_config = self.config.get("plot1", {})
        plot2_config = self.config.get("plot2", {})

        # Ensure save paths exist
        save_path_plot1 = self.ensure_save_path("plot1")
        save_path_plot2 = self.ensure_save_path("plot2")

        # Plot 1: Success rate by orbit type
        orbit_success_rate = data_falcon9.groupby("Orbit")['Class'].mean().sort_values()
        ax1 = orbit_success_rate.plot(kind='barh', 
                                      figsize=(plot1_config.get("figure_size", {}).get("width", 10),
                                               plot1_config.get("figure_size", {}).get("height", 6)),
                                      color=plot1_config.get("color", "green"))
        plt.ylabel(plot1_config.get("y_axis_title", "Orbit Type"))
        plt.xlabel(plot1_config.get("x_axis_title", "Success Rate"))
        plt.title(plot1_config.get("title", "SpaceX Success Rate by Orbit Type"))
        for container in ax1.containers:
            ax1.bar_label(container, fontsize=plot1_config.get("font_size", 15))

        # Save or display the first plot
        
        print("Displaying plot 1")
        plt.show()

        if save_path_plot1:
            plt.savefig(os.path.join(save_path_plot1, 'plot1.png'))
        
        plt.clf()  # Clear the figure for the next plot
        plt.figure()
        
        #Plot 2: Success rate over years
        years = [date.year for date in data_falcon9['Date']]
        df_yearly = pd.DataFrame({'year': years, 'Class': data_falcon9['Class']})
        ax2 = sns.lineplot(x=np.unique(years), y=df_yearly.groupby('year')['Class'].mean(),
        color=plot2_config.get("color", "blue"))
        plt.xlabel(plot2_config.get("x_axis_title", "Year"))
        plt.ylabel(plot2_config.get("y_axis_title", "Success Rate"))
        plt.title(plot2_config.get("title", "SpaceX Success Rate Over Years"))
        
        # Save or display the second plot
        print("Displaying plot 2")
        plt.show()
        if save_path_plot2:
            plt.savefig(os.path.join(save_path_plot2, 'plot2.png'))
        
        plt.clf()

        return plt.gcf()
