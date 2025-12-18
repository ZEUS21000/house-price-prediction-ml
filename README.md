# House Price Prediction (Machine Learning)

This project was built as part of my self-study in machine learning to better understand
data preprocessing, feature handling, and regression models using real tabular housing data.


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
- A Random Forest Regressor was chosen as a strong and reliable baseline for tabular data.
- One-hot encoding was used to safely handle categorical features.
- Median and most-frequent imputation were applied to deal with missing values.
- MAE and RMSE were selected as evaluation metrics for easier interpretation of model performance.

This project helped me understand the end-to-end machine learning workflow,
from data preprocessing to model training, evaluation, and saving trained models.

