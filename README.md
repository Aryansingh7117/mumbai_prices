# 🏠 House Price Prediction using Linear Regression

A complete Machine Learning project built using Python, demonstrating the full workflow from data cleaning to model training and evaluation using pandas, NumPy, and scikit-learn.

---

## 📖 Overview

This project predicts house prices based on features like BHK, area, region, and property status. It includes data preprocessing, encoding, normalization, and training a Linear Regression model to generate accurate price predictions.

---

## ⚙️ Steps Involved

### 1️⃣ Data Cleaning
- Removed missing or invalid entries.
- Converted all price values into Lakh.
- Selected key columns: `bhk`, `area`, `region`, `status`, and `price_lakh`.

---

### 2️⃣ Encoding
Converted categorical columns into numeric format using OneHotEncoding:

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[['type','status','age']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['type','status','age']))
df = pd.concat([df, encoded_df], axis=1).drop(['type','status','age'], axis=1)

