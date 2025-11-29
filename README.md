# Urban Heat Island Detection and Analysis - Hamburg Case Study

## Project Overview

This project focuses on the detection and analysis of Urban Heat Islands (UHIs) in Hamburg, Germany. UHIs are a critical environmental challenge where metropolitan areas experience higher temperatures than surrounding rural regions due to human activities and changes in land cover. This phenomenon leads to increased energy consumption, air pollution, and heat-related health issues, making its study crucial for sustainable urban planning. 

The core of this project is to leverage multi-source data, including satellite imagery and vector data, with advanced deep learning techniques (specifically U-Net architectures) for high-resolution UHI mapping and analysis. The study aims to understand the evolution of UHI intensity, its correlation with urban morphology and vegetation, and to develop a predictive model for future UHI patterns.

**Author:** Ozan Tuncbilek


## Key Objectives & Contributions

* **High-Resolution UHI Mapping:** Develop a methodology to map UHIs at a 10m/pixel resolution, improving upon traditional methods. 
* **Temporal Analysis:** Analyze the evolution of Hamburg's UHI intensity from 2014 to 2024, focusing on peak summer months. This includes selecting satellite imagery corresponding to the hottest meteorological days.
* **Correlation Analysis:** Investigate the spatial and statistical correlation between vegetation degradation (NDVI decline) and surface temperature increases in high-risk urban districts. 
* **Semantic Segmentation:** Employ a lightweight U-Net model, trained on fused Land Surface Temperature (LST), NDVI, and OpenStreetMap (OSM) derived features, to achieve high accuracy (target >90%) in detecting UHI zones in near-real-time. 
* **UHI Forecasting:** Develop a deep learning-based geospatial pipeline to forecast UHI distribution by 2030 under ongoing urbanization and vegetation loss trends. 
* **Policy Impact:** Provide data-driven insights to inform Hamburg's Urban Development Senate, particularly for targeted greening strategies. 
* **Commercial Potential:** Explore the development of a SaaS platform or API for UHI detection and heat risk assessment for urban planners, insurance companies, and real estate developers. 

## Methodology

The project employs a multi-faceted approach:

1.  **Data Acquisition:**
    * **Satellite Imagery:**
        * Landsat 8/9 (OLI/TIRS): For LST derivation and multispectral data. 
        * Sentinel-2 (MSI): For spectral indices like NDVI, NDBI, and NDWI at 10-20m resolution. 
        * MODIS: For broader LST context and potential temporal data fusion. 
    * **Vector Data:**
        * OpenStreetMap (OSM): Snapshots for 2015 and 2025 for buildings, roads, and green spaces. 
    * **Auxiliary Data:** Meteorological records, existing vegetation maps, and official land-use/land-cover datasets. 

2.  **Data Processing:**
    * Preprocessing of satellite data (radiometric calibration, atmospheric correction, reprojection, resampling to 10m). 
    * LST derivation from Landsat and MODIS, guided by meteorological records to identify hottest days. Landsat Collection 2 Level-2 Surface Temperature products will be used. 
    * Calculation of spectral indices (NDVI, NDBI, NDWI) from Sentinel-2. 
    * Rasterization of OSM vector layers (buildings, roads, green spaces) to 10m resolution. 
    * Feature engineering from OSM data (e.g., building density, road density, proximity to green spaces). 

3.  **Modeling & Analysis:**
    * **Semantic Segmentation:** A U-Net architecture will be trained using multi-channel input arrays (LST, NDVI, OSM masks/features) to define UHI zones, vegetation, and built structures. 
    * **Temporal Analysis:** Comparison of UHI patterns from 2015-2025 using scenes from the hottest days. 
    * **Forecasting:** Time-series models (ARIMA, Prophet, TCN, or LSTM) will be trained on historical UHI intensity trends to forecast patterns for 2030. 

4.  **Deployment & Application:**
    * Exploration of near-real-time UHI detection using trained models on cloud platforms (e.g., Google Earth Engine, AWS). 
    * Development of UHI maps, time-series visualizations, and predictions to support urban planning and climate adaptation. 

## Research Questions

1.  How has Hamburg's Urban Heat Island (UHI) intensity evolved from 2014 to 2024 during peak summer months, based on high-resolution satellite-derived LST data and urban morphological trends? 
2.  What is the spatial and statistical correlation between vegetation degradation measured via NDVI decline and surface temperature increases in high-risk urban districts, as derived from multi-source geospatial data? 
3.  Can a lightweight semantic segmentation model (e.g., U-Net), trained on fused LST, NDVI, and OSM-derived features, achieve over 90% accuracy in detecting UHI zones in near-real-time scenarios? 
4.  How effective is a deep learning-based geospatial pipeline in forecasting Urban Heat Island distribution by 2030 under ongoing urbanization and vegetation loss trends, with a case study on Hamburg?

## Technology Stack

* **Programming Language:** Python
* **Core Libraries:** GDAL, Rasterio, NumPy, osmnx
* **Deep Learning:** TensorFlow/Keras (for U-Net and other architectures like DeepLabv3+)
* **Geospatial Platforms:** Google Earth Engine (GEE) via `geemap` 
* **Time-Series Forecasting:** ARIMA, Prophet, TCN, LSTM
* **Version Control:** Git, GitHub 
* **Data Storage:** AWS S3 or Google Drive
* **Development Environment:** Google Colab (Jupyter notebooks) 
* **Visualization:** Mapbox, ipywidgets, potentially CesiumJS, Kepler.gl, ipyleaflet

## Future Enhancements

* Benchmark alternative deep segmentation architectures (e.g., DeepLabv3+, SegFormer). 
* Extend the study to multiple cities. 
* Integrate additional environmental variables. 


## License

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
