# IBEX Framework
**Author:** Jan Biały  

A Python framework for preprocessing, filtering, and analysis of energetic neutral atom (ENA) data collected by the NASA **IBEX (Interstellar Boundary Explorer)** mission.

---

## Description

The framework provides a modular pipeline for transforming raw IBEX-Hi and IBEX-Lo event data into analysis-ready datasets and for performing statistical and machine-learning-based studies of inter-channel correlations.

The core functionality is implemented in the `IBEX_Module` package and consists of the following components:

### Core modules

1. **`TensorCreator.py`**  
   Contains the `TensorCreator` class, responsible for the initial stage of data preprocessing.  
   It loads raw IBEX event files and generates PyTorch tensors that serve as the basis for further processing.

2. **`TensorAnalyzer.py`**  
   Contains the `ChannelAnalyzer` class, which performs advanced filtering and aggregation based on the instruction files  
   `HiCullGoodTimes.txt` and `LoGoodTimes.txt` provided with the IBEX data release.

   These instruction files describe time intervals in which valid ENA detections were recorded by the IBEX-Hi and IBEX-Lo instruments.

   The class supports multiple output modes:
   - **`_extract_good_raw_data`** – generation of filtered raw tensors restricted to valid time intervals,
   - **`_calculate_good_data_sums`** – computation of ENA count sums for each valid time range,
   - **`_aggregate_data_for_global_analysis`** – statistical aggregation (means, standard deviations) of ENA rates, incoming ENA direction parameters, and spacecraft position vectors (X, Y, Z) relative to Earth.

3. **`IBEX_NN.py`**  
   Contains additional data processing utilities and the `RateAutoencoder` class.  
   The autoencoder implements an undercomplete neural network architecture for dimensionality reduction and correlation analysis, along with training routines and automated visualization of reconstruction results.

4. **`FileMerger.py`**  
   Provides helper utilities for concatenating large intermediate files in cases where full tensor generation exceeds local hardware limitations.

5. **`main.py`**  
   Entry point for running the complete data processing, neural network training, and evaluation pipeline.

---

## Additional analysis

The repository also includes the Jupyter notebook **`IBEX_analysis.ipynb`**, which contains:
- exploratory data analysis (EDA),
- principal component analysis (PCA),
- channel ablation experiments using an MLP regressor model.

---

## Documentation

A detailed technical description of the framework, including API documentation generated directly from docstrings, is available in the following forms:

- **PDF documentation**:  
  `docs/ibexframework.pdf`

- **Online HTML documentation**:  
  *(link to be added after publication via GitHub Pages)*

---

## Requirements

The framework is written in Python and relies on standard scientific and machine learning libraries, including:
- NumPy
- PyTorch
- scikit-learn
- Matplotlib

---

## Notes

This framework was developed for scientific data analysis and research purposes and reflects the structure and characteristics of the publicly available IBEX data products.
