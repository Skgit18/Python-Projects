# Project - 1

This repository contains the code and analysis for a project focused on the spatio-temporal variations of 2-meter air temperature over the Indian subcontinent between 2014 and 2023. The project uses ERA5 reanalysis data and Python-based libraries for data manipulation, analysis, and visualization. This document provides an overview of the methods, analysis steps, and results.

## Project Overview

The project aims to explore surface air temperature variations in India through several techniques, including:
1. **Data Extraction and Preprocessing**:
   - We download ERA5 reanalysis data for 2-meter air temperature and specific humidity.
   - The data is a subset for India, covering the period from January 2014 to December 2023.
   - Only 2-meter air temperature is used for analysis.

2. **Anomaly Calculation**:
   - Monthly temperature anomalies are calculated by subtracting the climatological mean (for each month across the years) from the actual values. This approach preserves both spatial and temporal variability inherent in India’s diverse climatic regions.
   - Warm and cold anomalies are visualized to highlight periods of significant temperature deviations.

3. **Time-Series Decomposition**:
   - The temperature data is decomposed into three main components: trend, seasonal variations, and residuals.
   - This helps to separate long-term temperature trends from seasonal cycles and short-term irregular fluctuations.

4. **Spatial Visualization**:
   - The decadal mean temperature for each month is computed and visualized across the Indian subcontinent.
   - The spatial plots provide insights into the regional distribution of temperatures and highlight areas particularly prone to extreme conditions during different months.

5. **Future Directions**:
   - The next phase will involve higher-level computations, such as calculating temperature indices (e.g., cooling and heating degree days) and projecting future temperature patterns using statistical and machine learning models.
