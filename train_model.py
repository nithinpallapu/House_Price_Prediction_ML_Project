import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("indian_house_price_final.csv")
df.drop_duplicates(inplace=True)

y = df["Price_in_Lakhs"]
X = df.drop(columns=[
    "ID", "Locality", "Price_per_SqFt", "Year_Built",
    "Floor_No", "Owner_Type", "Availability_Status",
    "Price_in_Lakhs"
])

num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

preprocessor = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)
])

model = Pipeline([
    ("preprocess", preprocessor),
    ("dt", DecisionTreeRegressor(random_state=42))
])

params = {
    "dt__max_depth": [5, 10, None],
    "dt__min_samples_split": [2, 5, 10],
    "dt__min_samples_leaf": [1, 2, 5]
}

search = RandomizedSearchCV(
    model, params, n_iter=10, cv=3,
    scoring="r2", random_state=42
)

search.fit(X, y)

# SAVE MODEL IN SAME ENVIRONMENT
with open("house_price_model (1).pkl", "wb") as f:
    pickle.dump(search.best_estimator_, f)

print("Model trained and saved successfully!")
