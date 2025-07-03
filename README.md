
# Rainfall Prediction Using Machine Learning

Predicting whether it will rain tomorrow based on historical weather data from Australian cities.

---

## Author

**Mitali Gupta**  
GitHub: [mgupta004](https://github.com/mgupta004)  
LinkedIn: [linkedin.com/in/mgupta004](https://www.linkedin.com/in/mgupta004)

---

## Project Overview

Rainfall prediction is a vital part of weather forecasting that helps in planning for agriculture, transport, and public safety. This project uses real-world weather data to train and evaluate multiple machine learning models to predict if it will rain the next day.

---

## Dataset

- **Source**: [Kaggle - Rain in Australia](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)
- **Size**: 142,193 rows × 24 columns
- **Target**: `RainTomorrow` — whether it will rain the next day (Yes/No)

---

## Preprocessing Steps

1. **Class Imbalance Handling**: Oversampled minority class using `resample()`.
2. **Missing Values**:
   - Categorical: Filled with mode.
   - Numerical: Imputed using MICE (Multiple Imputation by Chained Equations).
3. **Outlier Detection**: Removed using Interquartile Range (IQR).
4. **Label Encoding**: Converted categorical features to numeric.
5. **Feature Scaling**:
   - MinMaxScaler (for chi-square filtering).
   - StandardScaler (for model training).

---

## Feature Selection

- **Filter Method**: Chi-Square test
- **Wrapper Method**: Random Forest importance

Top Features:
- Sunshine
- Humidity9am
- Cloud3pm
- Pressure9am
- RainToday
- RISK_MM

---

## Models Trained

- Logistic Regression
- Decision Tree
- Neural Network (MLP)
- Random Forest
- LightGBM
- CatBoost
- XGBoost
- Ensemble (Voting Classifier)

---

## Evaluation Metrics

- Accuracy
- ROC AUC
- Cohen's Kappa
- Confusion Matrix
- Execution Time

---

## Results

| Model             | Accuracy | ROC AUC | Cohen's Kappa | Notes                       |
|------------------|----------|---------|----------------|-----------------------------|
| Logistic Regression | Moderate | Moderate | Low           | Baseline model              |
| Decision Tree     | Moderate | Moderate | Moderate      | Interpretable               |
| Neural Network    | High     | High    | High          | Slower to train             |
| Random Forest     | High     | High    | High          | Good performance and speed  |
| LightGBM          | High     | High    | High          | Fast gradient boosting      |
| CatBoost          | Very High| Very High | Very High    | Best class boundaries       |
| XGBoost           | Best     | Best    | Best          | Highest overall performance |
| Ensemble          | High     | High    | High          | Combined multiple models    |

---

## Visualizations

- Correlation Heatmap
- ROC Curves
- Feature Importance
- Decision Boundaries using `mlxtend`

---

## How to Run

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost mlxtend
```

### Launch Notebook

```bash
jupyter notebook Rainfall_Prediction.ipynb
```

---

## Project Structure

```
Rainfall-Prediction/
├── Rainfall_Prediction.ipynb      # Main Jupyter Notebook
├── weatherAUS.csv                 # Dataset
├── README.md                      # Project overview and instructions

```

---



## Contributions

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## Acknowledgments

- Dataset by Australian Government Bureau of Meteorology
- Inspiration from TheCleverProgrammer.com article on Rainfall Prediction
