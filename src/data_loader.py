import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

url = "https://archive.ics.uci.edu/dataset/162/forest+fires"

response = requests.get(url)

print("Status Code:", response.status_code)

soup = BeautifulSoup(response.text, 'html.parser')
baslik = soup.find('h1')
if baslik:
    print("Veri Seti:", baslik.text.strip())
else:
    print("Veri seti bulunamadı.")

csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"

save_dir = os.path.join("data", "raw")
save_path = os.path.join(save_dir, "forestfires.csv")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Uyarı: '{save_dir}' klasörü bulunamadı, otomatik oluşturuldu.")

try:
    df = pd.read_csv(csv_url)
    df.to_csv(save_path, index=False)
    
    print(f"Veri seti indirildi ve '{save_path}' konumuna kaydedildi.")
    print(f"İndirilen Veri Boyutu: {df.shape[0]} satır, {df.shape[1]} sütun.")
except Exception as e:
    print(f"HATA: {e}")