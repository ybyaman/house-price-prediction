import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


model_path = os.path.join("model", "house_price_model.pkl")
test_data_path = os.path.join("data", "train.csv") 

if not os.path.exists(model_path):
    print("Hata: Önce train.py çalıştırılıp model kaydedilmelidir!")
else:
    model = joblib.load(model_path)
    df = pd.read_csv(test_data_path)

    
X = df.drop(["SalePrice", "Id"], axis=1)
y_true_log = np.log1p(df["SalePrice"])


y_pred_log = model.predict(X)


y_true = np.expm1(y_true_log)
y_pred = np.expm1(y_pred_log)


rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n--- MODEL DEĞERLENDİRME SONUÇLARI ---")
print(f"R2 Score: %{r2*100:.2f}")
print(f"Ortalama Hata (MAE): ${mae:,.2f}")
print(f"Kök Ortalama Kare Hata (RMSE): ${rmse:,.2f}")

  
print("\n--- İLK 5 TAHMİN ÖRNEĞİ ---")
comparison = pd.DataFrame({'Gerçek Fiyat': y_true, 'Tahmin Edilen': y_pred})
print(comparison.head())