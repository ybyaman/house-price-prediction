# 🏠 House Prices - Advanced Regression Techniques

Bu proje, Kaggle'ın popüler "House Prices" veri setini kullanarak konut fiyatlarını tahmin etmek amacıyla geliştirilmiş bir **Makine Öğrenmesi (Machine Learning)** projesidir. Proje; veri temizleme, özellik mühendisliği ve regresyon modellerini içeren uçtan uca bir pipeline yapısına sahiptir.

## 📋 Proje Özeti
Proje kapsamında, 79 farklı değişken (oda sayısı, metrekare, konum, yapım yılı vb.) analiz edilerek evlerin satış fiyatları (`SalePrice`) yüksek doğrulukla tahmin edilmektedir.

## 🏗️ Proje Yapısı
Proje modüler bir mimariyle kurgulanmıştır:

* **`data/`**: Eğitim (`train.csv`) ve test (`test.csv`) veri setlerini içerir.
* **`src/preprocess.py`**: Aykırı değerlerin temizlenmesi ve yeni özelliklerin (TotalSF, TotalBath vb.) türetilmesi.
* **`src/train.py`**: Modelin eğitilmesi, sayısal/kategorik verilerin işlenmesi ve modelin `.pkl` olarak kaydedilmesi.
* **`src/evaluate.py`**: Eğitilen modelin R2 Score ve RMSE gibi metriklerle performans analizi.
* **`src/predict.py`**: Fiyatı bilinmeyen test verileri üzerinde tahmin yürütme.
* **`model/`**: Eğitilmiş `house_price_model.pkl` dosyasının bulunduğu dizin.

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler
* **Dil:** Python 3.9+
* **Veri Analizi:** Pandas, Numpy
* **Makine Öğrenmesi:** Scikit-Learn (Random Forest Regressor, Pipeline, ColumnTransformer)
* **Model Saklama:** Joblib

## 🚀 Kurulum ve Çalıştırma

1. **Depoyu klonlayın veya indirin:**
   ```bash
   git clone [https://github.com/ybyaman/house-price-prediction.git](https://github.com/ybyaman/house-price-prediction.git)
   cd house-price-prediction
