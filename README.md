# 🍄 Mushroom Edibility Prediction

This project implements a machine learning model to classify mushrooms as **edible** or **poisonous** based on their physical characteristics. It uses the Logistic Regression algorithm and includes detailed data preprocessing steps due to the all-categorical nature of the dataset.

---

## 🎯 Project Goal

The primary goal is to build a highly accurate classification model capable of distinguishing between poisonous (`p`) and edible (`e`) mushrooms, minimizing the critical error of predicting a poisonous mushroom as edible (False Negative).

---

## 💾 Dataset

* **Source:** UCI Machine Learning Repository (commonly found as `mushrooms.csv`).
* **Size:** 8124 samples with 23 features.
* **Target Variable:** `class` (`p` for poisonous, `e` for edible).
* **Features:** All 22 predictor features are **categorical**.
* **Data Balance:** Initial exploration confirmed the dataset is fairly **balanced** between the two classes.

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn (sklearn)

---

## 💻 Methodology

The complete workflow involved loading, preprocessing, modeling, and evaluation.

### 1. Data Preprocessing

Since all features are categorical, the data required extensive encoding and standardization. A `ColumnTransformer` and `Pipeline` were used to streamline this process.

**Key Preprocessing Steps:**
1.  **Target Encoding:** The target variable `y` (`class`) was converted using **`LabelEncoder`** (Poisonous -> 1, Edible -> 0).
2.  **Feature Encoding:**
    * **One-Hot Encoding:** Applied to binary/low-cardinality features.
    * **Ordinal Encoding:** Applied to all other categorical features.
3.  **Scaling:** The resulting sparse matrix was scaled using **`StandardScaler`**.

**Code Snippet: Preprocessing Setup**

```python
# Separate features and target
X = df.drop(columns=['class'])
y = df['class']
```

# Encode target variable
```python
from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y)
```
# Define columns for different encoding types
```python
OneHot = ['cap-surface','bruises','gill-attachment','gill-spacing','gill-size','stalk-shape','veil-type','ring-number']
ordinal = ['cap-shape','cap-color','odor','gill-color','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-color','ring-type','spore-print-color','population','habitat']
```
# Create ColumnTransformer for preprocessing
```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 

preprocessor = ColumnTransformer(
    transformers = [
        ('onehot', OneHotEncoder(handle_unknown='ignore'), OneHot),
        ('ordinal', OrdinalEncoder(), ordinal)
    ],
    remainder='drop'
)
```
# Apply preprocessing and scaling via a Pipeline
```python
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False))
])

X_processed = pipeline.fit_transform(X)
```
