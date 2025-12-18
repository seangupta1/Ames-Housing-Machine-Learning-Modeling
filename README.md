# View The Notebook
GitHub’s notebook renderer can be difficult to read.
For the best viewing and interactive experience, view the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seangupta1/Ames-Housing-Machine-Learning-Modeling/blob/main/ames_housing_ml_pipeline.ipynb)

# Summary
The Ames Housing Dataset provides data from home sales in Ames Iowa between 2006 and 2010. Preprocessing and feature engineering was performed on the data followed by model hyperparameter tuning and final training. Finally, analysis was completed on the performance of the models.

# Introduction
The real estate market is complex, consisting of many patterns and trends that are ever changing. Real estate data such as the Ames Housing Dataset provide real world insight into what is going on within the market and the city. It can be time consuming and difficult for people to analyze this large amount of data and draw conclusions about it manually. Machine learning models can aid in the analysis and interpretation of this data. In this report we will discuss six machine learning models that were trained to help better understand this dataset. These models could be used by real estate agents, home sellers and buyers, and government officials.

# Dataset
The Ames Housing dataset contains nearly 3,000 instances of data with over 70 features. The data was collected between 2006 and 2010 in Ames Iowa. The dataset has many features that could be used for a machine learning model including:

- Year sold
- Year built
- Number of bathrooms
- Number of bedrooms
- Neighborhood
- Sale price
- etc.

# Data Preprocessing & Feature Engineering

## Duplicates
To ensure duplicate features are removed, a manual check and algorithmic check were performed. No duplicate features were found. To ensure duplicate instances were removed, an algorithmic check was performed locating 7 duplicate instances. These duplicate instances were dropped from the dataset.

## Missing Values
An algorithmic check was performed to detect features that were missing more than 30% of data. There were 6 features that were missing more than 30% of data. According to the 30% rule, we should drop these features. The features missing more than 30% of data were as follows:

| Feature        | Percent Missing |
|----------------|----------------|
| Alley          | 93.2%          |
| Mas Vnr Type   | 60.6%          |
| Fireplace Qu   | 48.5%          |
| Pool QC        | 99.6%          |
| Fence          | 80.5%          |
| Misc Feature   | 96.4%          |

Imputation was performed on features missing less than 30% of data.

- Several categorical basement features were missing data when the home had no basement. These features were `Bsmt Qual`, `Bsmt Cond`, `Bsmt Exposure`, `BsmtFin Type 1`, and `BsmtFin Type 2`. This was resolved by imputing a `NO_BSMT` category to fill missing values where the `Total Bsmt SF` is 0.  
- Several categorical garage features were missing data when the home had no garage. These features were `Garage Type`, `Garage Qual`, `Garage Cond`, and `Garage Finish`. This was resolved by imputing a `NO_GARAGE` category to fill missing values where the `Garage Cars` is 0.  
- The `Lot Frontage` feature was missing 16.7% of data. This was significant but still less than 30% missing so I wanted to keep this feature. My solution was to impute a 0 where the data is missing and create a binary feature `LotFrontage_missing` to show where this feature was missing.

## Unique Features
A manual and algorithmic check were performed to detect features that were unique per row. The features that were unique per row included `Order` and `PID`. These features were dropped from the dataset.

## Outliers
Outlier detection was performed on the training data. Outliers were detected in `Overall Qual`, `Mas Vnr Area`, and `Year Built`. These outliers were handled through imputing the mean, imputing the median, and dropping the value respectively.

## Encoding
For categorical features, they were encoded using ordinal and one hot encoding.  

- **Ordinal encoded features:** `Exter Qual`, `Exter Cond`, `Bsmt Qual`, `Bsmt Cond`, `Heating QC`, `Kitchen Qual`, `Garage Qual`, `Garage Cond`.  
- **One hot encoded features:** `MS Zoning`, `Street`, `Lot Shape`, `Land Contour`, `Utilities`, `Lot Config`, `Land Slope`, `Neighborhood`, `Condition 1`, `Condition 2`, `Bldg Type`, `House Style`, `Roof Style`, `Roof Matl`, `Exterior 1st`, `Exterior 2nd`, `Foundation`, `Bsmt Exposure`, `BsmtFin Type 1`, `BsmtFin Type 2`, `Heating`, `Central Air`, `Electrical`, `Functional`, `Garage Type`, `Garage Finish`, `Paved Drive`, `Sale Type`, `Sale Condition`.

## Scaling
All originally numerical features were scaled using the standard scaler from scikit-learn. The scaler object was fit on only the training data and then applied to both the training and testing data to avoid data leakage.

# Modeling
Models were made to solve three machine learning problems: classification, regression, and clustering.

## Classification
For classification, the problem is to predict the type of neighborhood the home is located in. The input features are `Lot Area`, `SalePrice`, `Lot Frontage`, and `Year Build`. The output feature is the type of neighborhood a home is in.  

Neighborhood labels are classified as follows:

| Category   | Neighborhoods |
|------------|----------------|
| Rural      | MeadowV, Greens, GrnHill, Landmrk |
| Urban      | IDOTRR, SWISU, OldTown, BrkSide |
| Suburban   | NAmes, CollgCr, Somerst, NridgHt, Gilbert, NWAmes, SawyerW, Mitchel, NoRidge, StoneBr, ClearCr, Blmngtn, NPkVill, Timber, Edwards, Sawyer, Crawfor |

The models used are **K-Nearest Neighbor (KNN)** and **Linear Discriminant Analysis (LDA)**. For each model, `GridSearchCV` is used with five folds to find the best hyperparameters.  

- KNN: `n_neighbors = 5`, `metric = Manhattan`  
- LDA: `shrinkage = 0.9`, `solver = lsqr`  

## Regression
For regression, the problem is to predict the sale price of the home. The input features are all numeric features including categorical features that were encoded. `SalePrice` was removed from the input data. The output feature is `SalePrice`.  

Models used:

- **LASSO** – Best hyperparameter: `alpha = 1000`  
- **Elastic Net** – Best hyperparameters: `alpha = 1`, `l1_ratio = 0.7`  

GridSearchCV with five folds was used for hyperparameter tuning.

## Clustering
For clustering, the problem is to cluster homes based on target sale market. The input feature `Total Flr SF` was engineered through the summation of `1st Flr SF`, `2nd Flr SF`, and `BSMT Total SF`. Input features are `SalePrice` and `Total Flr SF`.  

Two clusters were determined as ideal through elbow plot, silhouette vs. k figure, and silhouette plots.  

- **K-Means:** `n_clusters = 2`  
- **Agglomerative Clustering:** `linkage = ward`, `metric = euclidean`, `n_clusters = 2`

# Performance Summaries & Key Insights

## Classification
The following shows the report from the KNN and LDA classification models:  

KNN performed better than LDA, likely because KNN looks for the closest datapoints to make a prediction. Homes in the same region of the city have similar features.

## Regression

| Metric                   | LASSO   | Elastic Net |
|---------------------------|---------|-------------|
| Number of Features Used   | 81      | 252         |
| RMSE                      | 30627.61| 30627.61    |
| MAE                       | 17847.79| 17847.79    |
| R²                        | 0.883   | 0.881       |

LASSO had a slightly higher R² and used fewer features than Elastic Net.

## Clustering

| Clusters                 | K-Means | Agglomerative |
|---------------------------|---------|---------------|
| 0: Standard               | 1959    | 2254          |
| 1: High-end / Luxury      | 937     | 642           |

| Score                     | K-Means | Agglomerative |
|---------------------------|---------|---------------|
| Silhouette                | 0.54    | 0.55          |
| Inertia                   | 6202.61 | n/a           |

The Agglomerative model placed more homes in the standard cluster than K-Means and had a higher silhouette score.


# Data Availability
The dataset used in this project was provided by the course instructor and **cannot be publicly shared**.

As a result, the dataset is not included in this repository.  
The code, methodology, and analysis are provided to demonstrate the modeling approach and workflow.
