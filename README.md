# SpaceX Data Analyzer

A Python package for data analysis and information visualization based on the extracted data from the SpaceX API.

## Overview

SpaceX Data Analyzer is a Python package that facilitates data analysis and information visualization using data obtained from the SpaceX API. The package provides a variety of functionalities for exploring and interpreting data related to SpaceX missions, rockets, launches, and more.

## Features

- **Data Extraction:** Easily retrieve data from the SpaceX API with a few simple commands, making it hassle-free to keep your dataset up-to-date.

- **Data Analysis:** Perform in-depth analysis on various aspects of SpaceX missions, including booster versions, launch sites, payload data, and core information.

- **Information Visualization:** Create compelling visualizations to better understand and communicate the patterns and trends within the extracted data. The package supports a variety of chart types, including bar charts and line charts.

- **User-Friendly Interface:** The package comes with a straightforward interface, allowing users to navigate through the data effortlessly.

## Installation

```bash
pip install git+https://github.com/kamalakbari7/BSSApkg
```

## Getting Started

```python
import os

os.chdir('path/to/your/project/folder') # configs folder must be inside your project folder

from BSSApkg import Analysis
analysis = Analysis.Analysis('./configs/analysis_config.yml')

# Fetch the initial dataset
analysis.load_data()

# Perform computations or data processing
processed_data = analysis.compute_analysis()

# Visualize the results using the plot_data method
analysis.plot_data(processed_data)

```

## Configuration

The package utilizes configuration files (`system_config.yml`, `user_config.yml`, and `analysis_config.yml`) inside the configs folder to customize settings. Ensure the paths to these configuration files are correctly specified. The **configs** folder must be located inside your project folder. To sse the graphs, see the **plot** folder.

## Documentation

Detailed documentation is available [here](link-to-your-documentation), providing comprehensive information on installation, usage, configuration, and additional features.


## License

This project is licensed under the [MIT License](LICENSE), making it open and accessible to everyone.