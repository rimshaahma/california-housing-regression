
# California Housing Price Prediction: Regression Analysis

## Project Overview

This project is focused on predicting **median house values** in California using the **California Housing Dataset** from **scikit-learn**. We will use two types of regression models to predict housing prices:
- **Linear Regression**: Assumes a linear relationship between features and target values.
- **Polynomial Regression (Degree = 2)**: Captures non-linear relationships by fitting a higher-degree polynomial to the data.

The performance of the models will be evaluated using common regression metrics:
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R²)**

We will also generate visualizations to better understand the data and model performance, including:
- **Correlation Heatmap**: Displays the relationships between features and the target variable.
- **Predicted vs. Actual Values Scatter Plot**: Visualizes how well the model's predictions match the actual values.

## Objective

The primary goal of this project is to:
1. Use **Linear Regression** and **Polynomial Regression** to predict the **median house value** for California districts based on various features like average income, housing age, number of rooms, etc.
2. Evaluate and compare the performance of both models using RMSE, MAE, and R² metrics.
3. Visualize the relationship between features and the target variable.

## Dataset

This project uses the **California Housing Dataset** available from the `sklearn.datasets` module. It contains information for different districts in California, with the following key features:

- **Features**:
  - `longitude`: Longitude of the district.
  - `latitude`: Latitude of the district.
  - `housing_median_age`: Median age of the houses in the district.
  - `total_rooms`: Total number of rooms in the district.
  - `total_bedrooms`: Total number of bedrooms in the district.
  - `population`: Population of the district.
  - `households`: Number of households in the district.
  - `median_income`: Median income of the district’s residents.
  
- **Target**: 
  - `median_house_value`: The median house value for the district.

## Key Features

- **Models**:
  - **Linear Regression**: Assumes a linear relationship between the features and the target variable.
  - **Polynomial Regression**: Fits a polynomial (degree 2) to capture non-linear relationships.

- **Evaluation Metrics**:
  - **RMSE (Root Mean Squared Error)**: Measures the average magnitude of the errors. The lower the value, the better the model.
  - **MAE (Mean Absolute Error)**: Measures the average magnitude of the errors in the model's predictions. Lower values are better.
  - **R² (R-squared)**: Measures the proportion of variance in the target variable that is predictable from the features. The higher the R² value, the better the model.

- **Visualizations**:
  - **Correlation Heatmap**: Shows the correlations between features and the target variable, helping to understand relationships in the data.
  - **Predicted vs Actual Values Scatter Plot**: Helps to visually compare the predicted house values against the actual values from the dataset.

### File Descriptions:
- **`california_housing.ipynb`**: This is the Jupyter notebook where the main analysis happens. It includes data loading, preprocessing, model training, evaluation, and visualizations.
- **`plots/`**: A folder where the generated visualizations (scatter plot and heatmap) are stored.
- **`requirements.txt`**: Contains a list of Python packages required to run the project.
- **`LICENSE`**: Provides details about the license under which the project is shared.

## How to Run the Project

Follow these steps to set up and run the project:

### 1. Clone the Repository

To clone the repository to your local machine, use the following command:

```bash
git clone https://github.com/your-username/california-housing-regression.git
cd california-housing-regression
```

### 2. Install Dependencies

The project requires several Python libraries. To install them, it's recommended to use a **virtual environment**. You can use `pip` to install the dependencies.

- First, install the required libraries by running:

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

Once the dependencies are installed, open the Jupyter notebook with the following command:

```bash
jupyter notebook california_housing.ipynb
```

This will open the notebook in your web browser, where you can run the code cells.

### 4. Run the Code

The Jupyter notebook contains the entire project workflow:
- **Data Loading**: The dataset is loaded and inspected.
- **Data Preprocessing**: Missing values are handled, and features are scaled.
- **Model Training**: Both Linear and Polynomial Regression models are trained.
- **Model Evaluation**: The models are evaluated using RMSE, MAE, and R².
- **Visualization**: Key visualizations are generated, including the correlation heatmap and predicted vs actual values plot.

## Example Output

### Model Evaluation:

| Model                | RMSE  | MAE   | R²   |
|----------------------|-------|-------|------|
| **Linear Regression** | 0.75  | 0.50  | 0.65 |
| **Polynomial Regression** | 0.65 | 0.42 | 0.78 |

- **Linear Regression** had a slightly higher RMSE and MAE than **Polynomial Regression**, but both models showed reasonable performance.
- **Polynomial Regression** outperformed **Linear Regression** in terms of R², indicating it explains more variance in the data.

### Data Visualizations:

1. **Correlation Heatmap**:
   This heatmap shows the correlations between different features and the target variable (median house value). The darker colors represent stronger correlations.

   ![Correlation Heatmap](_heatmap.png)

2. **Predicted vs Actual Values**:
   A scatter plot showing how well the predicted values from the models compare with the actual values in the dataset. Ideally, the points should lie close to the diagonal line.

   ![Predicted vs Actual Scatter Plot](actual.png)

## Data Preprocessing

### Steps:
1. **Handling Missing Values**: Missing values in the dataset were handled using **mean imputation** (replacing missing values with the feature mean).
2. **Feature Scaling**: The features were standardized using **StandardScaler** to bring them to a similar scale, which helps improve the performance of regression models.

## Requirements

The project requires the following Python libraries:

- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `jupyter`

You can install the necessary dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Topics Involved

This project covers the following topics:

- **linear-regression**
- **polynomial-regression**
- **model-evaluation**
- **rmse**
- **mae**
- **r2**
- **data-preprocessing**
- **data-visualization**
- **heatmap**
- **scatter-plot**
- **machine-learning**
- **scikit-learn**
- **pandas**
- **numpy**
- **jupyter-notebooks**
- **predictive-modeling**
- **regression-analysis**
- **data-science**

