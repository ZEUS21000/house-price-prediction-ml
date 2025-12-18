# House Price Prediction (Machine Learning)

This project uses a machine learning regression model to predict house prices from tabular housing data.

## Technologies
- Python
- Pandas
- scikit-learn
- Random Forest Regression

## Project Structure
- `train.py` – trains the ML model
- `requirements.txt` – project dependencies
- `model.joblib` – saved trained model (generated after training)

## How to run
1. Install dependencies:
```bash
pip install -r requirements.txt
Note: `data.csv` and `model.joblib` are not included in the repository. 
The dataset should be provided locally and the model file is generated after training.
## Design Choices
- Random Forest was chosen as a strong baseline model for tabular data.
- One-hot encoding was used to handle categorical features safely.
- MAE and RMSE were selected as evaluation metrics for interpretability.
