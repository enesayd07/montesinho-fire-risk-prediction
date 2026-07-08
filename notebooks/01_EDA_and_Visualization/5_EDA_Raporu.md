# MONTESİNHO ORMAN YANGINLARI: KEŞİFÇİ VERİ ANALİZİ (`EDA`) VE GÖRSELLEŞTİRME RAPORU

**Kapsam:** Portekiz Montesinho Tabiat Parkı Tarihsel Veri Seti (`2000-2003, Cortez & Morais, 2007`)  
**Metodoloji:** Ormancılık Fiziği, İstatistiksel Hipotez Testleri ve Uç Değer Teorisi (`Pop et al., 2026`)  
**Durum:** Keşifçi Veri Analizi Raporu

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 1. ÖZET
Montesinho Tabiat Parkı veri seti (`517 satır, 13 değişken`), **doğrusal olmayan (`Non-Linear`)**, **ağır kuyruklu (`Heavy-Tailed`)** ve **eşik bağımlı (`Threshold-Driven`)** bir yapıya sahiptir. 

6 adet makaleyi kaynak olarak kullanan bu 4 dosyalık proje, incelenen verilerin özelliklerine göre 4 farklı dosyaya bölünmüştür. İlk dosya temel bir bakış atmıştır. İkinci dosya zaman aralıklarının ve tarihsel değişkenlerin yangınlara etkisinin yanı sıra insan faktörünü de inceler. Üçüncü dosya yangınların çıktığı bölgeleri mekansal ve topografik olarak inceleyip diğer değişkenlerle analizini yapar. Dördüncü ve son dosya da meteorolojik değişkenler ile birlikte Kanada Orman Yangını Hava Endeksi değerlerini analiz eder.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 2. DOSYALARIN GENEL İÇERİĞİ

### *Dosya:* `01_Veri_Denetimi_ve_On_İnceleme.ipynb`

#### Yapısal Kusursuzluk ve Sıfır Kayıp Veri
* Veri seti 517 gözlem ve 13 değişkenden oluşmakta olup, eksik/kayıp değer (`Null/NaN`) oranı **%0.0'dır**. Tüm veri türleri doğru kategorik ve sayısal formatlara oturtulmuştur.

#### Sıfır Hasar Yığılması (`%47.8 - 247 Gün`) ve İnsani Müdahale Varlığı
* Hedef değişkenimiz olan `area` (yanan alan hektarı) incelendiğinde, 517 kaydın tam **247 tanesinde (`%47.8`)** yanan alanın `0.00 hektar` olduğu saptanmıştır.
* **Yani:** Sıfır değeri bir "eksik veri (`Missing Data`)" değil, tutuşma (`Ignition`) gerçekleşmesine veya ihbar girilmesine rağmen **itfaiyenin erken müdahalesiyle 100 m² (`0.01 ha`) altında söndürülen** veya büyümeden bastırılan başarılı insan vetolarını temsil eder.

#### Uç Değer Çarpıklığı (`Skewness: 12.85`) ve Dönüşüm Zorunluluğu
* `area` değişkeninin çarpıklık katsayısı **12.85**, basıklık (`Kurtosis`) değeri ise **278.88** gibi ekstrem düzeydedir. Bu durum, veri setindeki birkaç mega yangının (`1090.84 ha ve 746.28 ha`) tüm aritmetik ortalamayı yuttuğunu gösterir.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### *Dosya:* `02_Mevsimsel_ve_İnsani_Etkiler.ipynb`

#### Ağustos-Eylül Yoğunlaşması 
* Parktaki yangınların **%67'si (`345 kayıt`)**, toplam tahribatın ise **%86'sı (`5720+ hektar`)** yalnızca Ağustos ve Eylül aylarında gerçekleşmektedir.
* **Sebebi:** İlkbaharda ve haziranda yağan yağmurların etkisi yaz sonuna doğru tamamen tükenmekte; derin toprak kuraklığı endeksi (`DC`) 600 puanı aşarak orman tabanını yangın için uygun bir konuma getirir.

#### Antropojenik Mobilizasyon 
* **Hafta Sonu (`Sat-Sun`):** Piknikçiler, kampçılar ve doğa turizmi nedeniyle yangın çıkış frekansı (`ihbar sayısı`) en yüksek seviyededir. Ancak turistik bölgeler yollara yakın olduğu için itfaiye hızlı müdahale etmiş ve ortalama tahribat nispeten kontrol altında tutulmuş olabilir.
* **Hafta İçi (`Mon-Fri`):** Özellikle Çarşamba ve Perşembe günleri yangın başına düşen ortalama tahribat zirve yapmaktadır. Bunun nedeni, hafta içi ormanın iç ve uzak bölgelerinde yapılan **tarımsal anız yakma ve ormancılık faaliyetlerindeki** ihmaller ile hafta içi gözlemci sayısının azlığından dolayı müdahalenin gecikmesi olabilir.
* Ancak genel olarak hafta sonu insanların parka olan ziyaretinin artmasından dolayı her dönem hafta sonu yangın daha fazladır.

#### Sıcaklık-Nem-Yangın İlişkisi
* İlkbahar aylarında (`Mart`) düşük sıcaklığa (`~8-12°C`) rağmen rüzgar ve düşük nemle tetiklenen orta ölçekli yangınlar görülürken; yaz sonuna (`Ağustos/Eylül`) gelindiğinde sistem hem yüksek sıcaklık (`>25°C`) hem de kronik kuraklıkla tam bir afete dönüşmektedir.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### *Dosya:* `03_Mekansal_ve_Topografik_Yogunluk.ipynb`

#### `9x9` Grid Haritası 
* En büyük iki yangın merkezde ve birbirine aşırı yakın bölgelerde çıkmıştır. Veri setindeki konumlama 9x9'luk 81 adet bölgeye bölündüğü için analiz etmesi daha kolaydır.
* **(`[X=8, Y=6]` vb. noktalar):** Yangın çıkış frekansının en yüksek olduğu (`54 ihbar`) ancak tahribatın daha küçük kaldığı insan erişimine açık bölgelerdir.

#### Tekil Satır Yanılsaması 
* Literatürdeki tüm çalışmalar (`Cortez & Morais, 2007`), veriyi satır satır incelediği için bir gün içindeki toplam yıkımı görememiştir. Biz satırları **484 gerçek takvim gününe** topladığımızda (`Aggregation`), aynı gün içinde farklı koordinatlarda başlayan eşzamanlı yangınların birleşerek nasıl katlanarak büyüdüğünü ortaya çıkardık.

#### Gizli Mega-Afet Günlerinin Keşfi
* Satır bazında bakıldığında `50 hektar` altındaki "orta ölçekli" görülen 3 özel gün (`[182: DEC-MON 22 ha]`, `[272: MAR-MON 36 ha]` ve `[284: MAR-SAT 28 ha]`), aynı saatte çıkan diğer odaklar toplandığında gerçekte **`60.38 ha, 64.20 ha ve 57.32 ha`** büyüklüğünde tam birer **Gizli Mega-Afete** dönüşmüştür!

#### Çoklu Odaklı Günlerin Nedeni
* Parkta günde 2 veya daha fazla yangının çıktığı `29 çoklu odak gününde`, rüzgar hızı tekli günlere (`455 gün`) kıyasla **%15 artarak `4.52 km/h`** seviyesine çıkmış, nem %42.4'e inmiştir.
* Günlük birleşik orman yıkımı ise tekli günlere kıyasla **%21.4 zıplayarak ortalama `16.45 hektara`** fırlamıştır. Ayrıca koordinatlar arası mesafe analizinde (`2.5 birim eşiği`), kilometrelerce (`7.21 birim`) uzakta çıkan yangınların rüzgarla sıçrama değil, **yaygın insani ihmal/teyakkuz günleri** olduğu görülmüştür.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### *Dosya:* `04_Meteoroloji_ve_Kanada_FWI_Sistemi.ipynb`

#### Yağmur Veto Mekanizmasının İstatistiksel Gösterimi (`Welch t-testi & Fisher exact`)
* Hiç yağmur yağmayan `509 günde` yangın çıkma olasılığı **%52.7** iken, yağmur yağan `8 günde` bu oran **%25.0'a** çakılmıştır.
* **%75 Veto Başarısı:** Yağmurlu 8 günün 6'sında (`Index 243, 500, 501, 502, 3 ve 286`) yanan alan tam olarak `0.00 hektardır`. İnce yanıcı nemi (`FFMC`) üzerindeki Welch t-testi (`p = 0.0086`), yağmurun yanıcıları fiziksel olarak ıslatıp tutuşmayı bloke ettiğini %99 güvenle tescillemiştir.
* **Yaz Buharlaşma İstisnaları:** Yağmura rağmen yangın çıkan 2 Ağustos gününde (`10.82 ha ve 2.17 ha`), sıcaklığın `21°C buharlaşma eşiğinin` çok üstünde (`27.3°C ve 21.1°C`) olduğu saptanmıştır. Kızgın güneş altında yağan kısa süreli sağanak (`6.4 mm`), daha toprak altına (`DC`) ulaşamadan ağaç tepelerinden buharlaşmış ve rüzgarla yangın çıkabilmiştir.

#### Çifte Kuraklık Eşiği (`Double Drought Threshold: FFMC >= 88 & DC >= 500`)
* Hem ince yüzey yanıcılarının (`FFMC >= 88`) hem de derin kök/toprak tabakasının (`DC >= 500`) aynı anda kuruduğu **4. Bölge (Çifte Kuraklık Alanı)**, parktaki toplam `6642 hektarlık` yıkımın tam **`5162.21 HEKTARINI` (%77.7'si!)** tek başına gerçekleştirmiştir.

#### Tehlikeli Kesişim Eşiği (`temp >= 20°C & RH <= %40`)
* Zamanın sadece %31'ini (`160 gün`) oluşturan bu sıcak-kuru hava penceresinde, yangın çıkma olasılığı **%60.0'a**, yangın başına ortalama tahribat ise serin günlere kıyasla **%78 artarak `34.29 hektara`** çıkmaktadır. Toplam tahribatın yarısı (`3291 ha`) sadece bu 160 günde yaşanmıştır.

#### Orta Rüzgar Problemi 
* En yüksek ortalama tahribat (`32.23 ha`) fırtınalı günlerde değil, **orta şiddetli rüzgarlarda (`3.5-6.0 km/h`)** gerçekleşmiştir. Parktaki en büyük 2 mega-afet bu banttadır.
* **Fiziksel Neden:** Bu rüzgar bandı, insanların piknik/tarım yapmak için ormana gitmeye devam ettiği ("rahat hava" -> kıvılcım var) ve aynı zamanda çıkan kıvılcımı tepe yangınına çevirecek oksijen itişini (`ISI`) sağlayan ölümcül tatlı noktadır (`Sweet Spot`).
* **Fırtına Vetosu (`>6.0 km/h -> 11.94 ha`):** Fırtınalı günlerde tahribat ortalamasının düşmesi rüzgarın sönümlemesinden değil, olumsuz hava nedeniyle insanların ormana girmeyip kıvılcım çıkarmamasından (`Ignition Veto`) kaynaklanır. Nitekim o günlerde kıvılcım çıkarsa (`Konsol: APR 9.4 km/h`), rüzgar yangını `61.13 hektara` kadar büyütmektedir.

#### Pearson Korelasyon Sıfırlığı ve VIF Çoklu Doğrusallık Denetimi
* Isı haritasında `area` ile hiçbir meteorolojik değişkenin doğrusal korelasyonu çıkmamıştır (`maksimum |r| = 0.10`). Bu durum, klasik doğrusal modellerin (`OLS, Ridge, Lasso`) bu veri setinde kesinlikle çuvallayacağının matematiksel ispatıdır.
* VIF değerlerinin tamamı `10.0` eşiğinin altındadır. `DMC` ile `DC` arasındaki güçlü ilişki (`r = 0.68`) derinlik farkını temsil ettiği için her iki endeks de modelde tutulacaktır.

#### Pareto Uç Değer Teorisi (`Extreme Value Theory`)
* Tahribat dağılımı ağır kuyruklu (`Heavy-Tailed`) bir Pareto eğrisi sergilemektedir:
  * **Top %1 (`5 Mega Yangın`):** Toplam yıkımın **%38.08'ini (`2529.47 ha`)** tek başına yapmıştır.
  * **Top %10 (`51 Yangın`):** Toplam yıkımın tam **%80.12'sini (`5321.54 ha`)** gerçekleştirmiştir! (`80/20 kuralı burada 80/10 olarak gerçekleşmiştir`).
  * **Geriye Kalan %90 (`467 Küçük Yangın`):** Tüm yıkımın sadece `%19.9'undan` sorumludur.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Kaynak Makaleler:

** A Data Mining Approach to Predict Forest Fires using Meteorological Data : https://www.researchgate.net/publication/238767143_A_Data_Mining_Approach_to_Predict_Forest_Fires_using_Meteorological_Data

** Integrated AI Approaches for Forest Fire Prediction: A Comparative
Study of Regression and Deep Learning Models : https://www.scitepress.org/Papers/2026/142678/142678.pdf

** Forest Fire Prediction with Imbalanced Data Using a Deep Neural Network Method : https://www.researchgate.net/publication/362087598_Forest_Fire_Prediction_with_Imbalanced_Data_Using_a_Deep_Neural_Network_Method

** Predictive Modelling of Burned Area in the Montesinho Natural Park: A 
LightGBM / Random Forest Comparison with a Reactive Web Interface 
for Fire Risk Scenario Simulation : https://www.researchgate.net/publication/407136409_Predictive_Modelling_of_Burned_Area_in_the_Montesinho_Natural_Park_A_LightGBM_Random_Forest_Comparison_with_a_Reactive_Web_Interface_for_Fire_Risk_Scenario_Simulation_Watch_the_Video_Abstract_System_D

** Tackling the Wildfire Prediction Challenge: An Explainable Artificial Intelligence (XAI) Model Combining Extreme Gradient Boosting (XGBoost) with SHapley Additive exPlanations (SHAP) for Enhanced Interpretability and Accuracy : https://www.researchgate.net/publication/390859968_Tackling_the_Wildfire_Prediction_Challenge_An_Explainable_Artificial_Intelligence_XAI_Model_Combining_Extreme_Gradient_Boosting_XGBoost_with_SHapley_Additive_exPlanations_SHAP_for_Enhanced_Interpretab

** Wildfire Prediction Model Based on Spatial and Temporal Characteristics: A Case Study of a Wildfire in Portugal’s Montesinho Natural Park : https://www.researchgate.net/publication/362716881_Wildfire_Prediction_Model_Based_on_Spatial_and_Temporal_Characteristics_A_Case_Study_of_a_Wildfire_in_Portugal's_Montesinho_Natural_Park

