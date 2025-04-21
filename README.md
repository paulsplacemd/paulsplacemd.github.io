# Pauls Place Project
## Overview

This project and our geospatial project [link](https://github.com/paulsplacemd/Geospatialproject.git) aim to identify and understand healthcare issues and its un-accessibility in Southwest Baltimore using data science techniques. By analyzing spatial disparities, environmental factors, and community health outcomes, we aim to provide insights that can support data-driven interventions and equitable resource distribution for underserved communities. This work is in collaboration with Dr. Megan, the Program Lead for Health and Wellness at Paul’s Place.
This specific part of the project focuses on creating a general health risk predictive model using factors contributing to health and wellness within Southwest Baltimore. The model is currently a work in progress as we continue to collaborate with Paul's Place to gather more comprehensive community data for improvement.

## Project Specific Key Features

- Integrates demographic, environmental, and health-related datasets.
- Develops a general health risk predictive model based on key indicators.
- Identifies potential areas at high risk of general health and wellness deficit.
- Aims to support data-driven decisions for resource allocation and service delivery.


## Getting Started

### Data Sources

We utilized the following publicly available datasets:

- **American Community Survey (ACS) from the U.S. Census Bureau:** Provided socio-economic data such as income, education, health insurance, race, and age. ([https://www.census.gov/programs-surveys/acs/news/data-releases/2023.html](https://www.census.gov/programs-surveys/acs/news/data-releases/2023.html))
- **Homeless Shelter Locations from Open Baltimore:** Provided geospatial data on the locations of homeless shelters in Baltimore. ([https://data.baltimorecity.gov/datasets/baltimore::homeless-shelters-2/explore](https://data.baltimorecity.gov/datasets/baltimore::homeless-shelters-2/explore))
- **Health Department Dataset (Baltimore City):** Included information on store density (alcohol, tobacco, grocery), mortality, substance abuse, and more. ([https://data.baltimorecity.gov/datasets/e37ce649df4344dab174b34593b1c4b6_0/explore?location=39.307459%2C-76.628697%2C11.39&showTable=true](https://data.baltimorecity.gov/datasets/e37ce649df4344dab174b34593b1c4b6_0/explore?location=39.307459%2C-76.628697%2C11.39&showTable=true))

### Installation

- Clone the repository:
    ```bash
    git clone [repository URL]
    ```
- Ensure you have the necessary Python libraries installed:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    # For potential future geospatial analysis:
    # pip install geopandas plotly
    ```
- Download the datasets mentioned in the "Data Sources" section and ensure the file paths in the code match their locations.

## Utilization

This snippets show the initial steps in data exploration, cleaning, preprocessing, feature engineering, and model development.

- **Data Loading:** The Python code shows how to load the liquor store density, healthy food index, and crime rate datasets using pandas.
- **Data Cleaning and Preprocessing:** Missing values were handled, and feature selection was performed to focus on relevant variables for the health index score. Datasets were merged using zip codes and neighborhood names.
- **Feature Engineering:** Health-related score indices ('Healthy\_Food\_Score', 'Crime\_Score', 'Uninsured\_Score', 'Education\_Score', 'Liquor\_Store\_Score', 'Income\_Score') were created based on predefined thresholds.
- **Data Transformation:** These health indices were weighted equally to create a 'Composite\_Score\_Weighted'.
- **Exploratory Data Analysis:** Visualizations were generated to understand the relationships between variables (though the specific visualizations are not shown in the snippets).
- **Model Development:** Logistic Regression and K-Nearest Neighbors (KNN) were selected as initial models.
- **Model Training:** The data was split into training 70% and test 30% sets, and features were scaled using `StandardScaler`.
- **Model Evaluation:** Performance metrics such as Accuracy, Precision, Recall, and F1-Score were used, along with a Confusion Matrix. Initial perfect scores on the small test set warranted caution due to potential overfitting.
- **Hyperparameter Tuning:** `GridSearchCV` was used for KNN, with the best result at k=1. Minimal tuning was done for Logistic Regression.
- **Cross-Validation:** 5-fold cross-validation highlighted potential instability due to the small dataset size.

To run the provided code snippets, ensure your Python environment is set up with the necessary libraries and that the file paths to the datasets are correct. You can execute these in a Jupyter Notebook or a Python script.


Contributions to this project are welcome.

[Project Markdown File](https://github.com/paulsplacemd/paulsplacemd.github.io/blob/1a76cd236a01e9974797d99d51558ef8f87c56cb/predictivemodel_paulsplace-3/predictivemodel_paulsplace.md)


## Contact

For questions or further information, please contact:

# Howard University Pauls Place Health Squad.
[Email](paulsplacemd@gmail.com)
