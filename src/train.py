import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.preprocess import clean_outliers, feature_engineering
from preprocess import clean_outliers, feature_engineering

train_path = os.path.join("data", "train.csv")
df = pd.read_csv(train_path)

df = clean_outliers(df)
df = feature_engineering(df)

X = df.drop(["SalePrice", "Id"], axis=1)
y = np.log1p(df["SalePrice"]) 


numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Model eğitiliyor, lütfen bekleyin...")
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Başarı Skoru (R2): {r2_score(y_test, y_pred):.4f}")
print(f"Hata Payı (RMSE): {rmse:.4f}")


os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/house_price_model.pkl")
print("Model 'model/house_price_model.pkl' olarak kaydedildi!")