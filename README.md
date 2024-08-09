# Unsupervised Risk Factor Finding

This project aims to identify common risk factors in financial data using unsupervised learning techniques. By clustering correlated financial signals, we can better understand the underlying factors that drive market behavior.

## Table of Contents

- [Overview](#overview)
- [Data](#data)
- [Methods](#methods)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Overview

In the world of finance, understanding the factors that affect market behavior is crucial for risk management and investment strategies. This project utilizes machine learning techniques to discover these factors without prior labeling, focusing on financial signals and their relationships.

## Data

The primary data file used in this project is `PredictorPortsFull.csv`. This dataset includes:

- **Signal Names:** Identifiers for different financial signals.
- **Returns (ret):** Historical return data for each signal.
- **Port:** Portfolio information, filtered to `'LS'` (Long-Short) for this analysis.
- **Date:** Time index for the return data.

## Methods

The project employs the following methods:

1. **Correlation Analysis:** Calculate the correlation between different financial signals based on historical returns.
2. **Distance Transformation:** Convert the correlation matrix into a distance matrix using a custom formula.
3. **Dimensionality Reduction:** Use Multi-Dimensional Scaling (MDS) to reduce the distance matrix to a 2D representation.
4. **Clustering:** Apply K-means clustering to identify natural groupings of signals, using both the Elbow and Silhouette methods to determine the optimal number of clusters.
5. **Cluster Sorting:** Sort cluster members by their distance to the cluster centroid for better interpretation.

## Usage

To run the analysis, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/unsupervised_risk_factor_finding.git
   cd unsupervised_risk_factor_finding
   ```

2. **Install Dependencies:**

   Ensure you have all necessary Python libraries installed (see [Dependencies](#dependencies) section).

3. **Run the Script:**

   ```bash
   python analysis_script.py
   ```

4. **View Results:**

   The script will print sorted clusters and display a plot of the 2D representation with cluster assignments.

## Results

The project outputs include:

- **Sorted Clusters:** Financial signals sorted by proximity to their cluster centroids.
- **Visualization:** A scatter plot showing the 2D representation of signal names with cluster assignments.

## Dependencies

This project requires the following Python libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

You can install these using pip:

```bash
pip install pandas numpy scikit-learn matplotlib
```
