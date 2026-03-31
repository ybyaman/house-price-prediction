import pandas as pd
import numpy as np
import joblib
import os
from preprocess import feature_engineering

# 1. MODELİ VE TEST VERİSİNİ YÜKLE
model_path = os.path.join("model", "house_price_model.pkl")
test_path = os.path.join("data", "test.csv")

if not os.path.exists(model_path):
    print("Hata: Model bulunamadı! Önce train.py çalıştırın.")
else:
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)
    test_ids = test_df['Id'] # Sonuç dosyası için Id'leri saklıyoruz

    # 2. ÖN İŞLEME
    # Eğitimde yaptığımız yeni özellikleri test verisine de ekliyoruz
    test_df = feature_engineering(test_df)
    
    # Gereksiz sütunları çıkar (Model sadece eğitimdeki X sütunlarını bekler)
    X_test = test_df.drop(["Id"], axis=1)

    # 3. TAHMİN YAP
    print("Test verisi için tahminler yapılıyor...")
    y_pred_log = model.predict(X_test)
    
    # Logaritmik değerleri gerçek fiyatlara çevir
    y_pred = np.expm1(y_pred_log)

    # 4. SUBMISSION DOSYASI OLUŞTUR
    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": y_pred
    })

    submission.to_csv("submission.csv", index=False)
    print("Tebrikler! 'submission.csv' dosyası oluşturuldu. Kaggle'a yüklemeye hazır.")