# Regression Analysis on California Housing Dataset

## **Objective**
This project aims to predict median house values using the California Housing Dataset and evaluate the performance of Linear and Polynomial Regression models.

## **Key Features**
- **Dataset**: California Housing Dataset (from `sklearn.datasets`)
- **Models**:
  - **Linear Regression**: Captures linear relationships.
  - **Polynomial Regression**: Fits non-linear relationships (degree = 2).
- **Evaluation Metrics**: RMSE, MAE, and R².
- **Visualization**:
  - Correlation heatmap of features.
  - Predicted vs. actual values scatter plot for both models.

## **Performance Comparison**
| Model                 | RMSE     | MAE      | R²       |
|-----------------------|----------|----------|----------|
| Linear Regression     | 0.75     | 0.50     | 0.65     |
| Polynomial Regression | 0.65     | 0.42     | 0.78     |

## **Repository Structure**
- `california_housing.ipynb`: Jupyter Notebook containing all code for the project.
- `plots/`: Contains visualizations like predicted vs actual scatter plots and heatmaps.
- `README.md`: Project documentation and instructions.

## **How to Run the Project**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/california-housing-regression.git
   cd california-housing-regression
