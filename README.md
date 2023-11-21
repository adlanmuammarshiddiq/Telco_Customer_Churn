# Capstone 3 : Telco Customer Churn Prediction

## Bussines Understanding
**Context :**
Pada saat ini, industri telekomunikasi telah dengan berkembang sangat pesat. Banyak sekali perusahaan telekomunikasi yang bersaing menawarkan jasanya menggunakan sistem berlangganan untuk menjual jasanya, sehingga persaingan antar perusahaan semakin ketat. Salah satu tantangan yang kini dihadapi oleh perusahaan adalah mempertahankan jumlah pengguna customer yang telah berlangganan agar tidak beralih ke perusahaan kompetitor.

Suatu perusahaan ingin mengetahui pelanggan yang bagaimana yang akan pindah (churn) dari perusahaan tersebut sehingga jumlah pelanggan yang beralih (churn) dapat dikurangi. Seorang Data Scientist diminta untuk membuat model prediksi yang tepat untuk menentukan pelanggan akan yang akan berhenti menggunakan layanan (churn) atau tidak dengan menggunakan Machine Learning.

Target :

0 : Tidak berhenti menggunakan layanan

1 : Berhenti menggunakan layanan (churn)

**Problem Statement :**
Untuk suatu perusahaan telekomunikasi, tingginya persentase pelanggan yang melakukan *churn* adalah salah satu indikator yang menjadi tingkat kegagalan suatu perusahaan tersebut, sehingga diperlukan upaya untuk mengurangi persentase pelanggan *churn* tersebut. Pada umumnya perusahaan lebih memilih untuk mempertahankan pelanggan, karena biaya untuk mempertahankan pelanggan *(customer retention cost)* lebih rendah daripada memperoleh pelanggan baru *(customer acquisition cost)*.Berdasarkan informasi dari internet, memperoleh pelanggan baru dapat menghabiskan biaya lima kali lebih banyak daripada mempertahankan pelanggan yang sudah ada. Adapun rata-rata biaya customer acquisition cost untuk industri telekomunikasi adalah sekitar $315 per new customer.

Salah satu cara perusahaan telekomunikasi mempertahankan para customernya agar tetap berlangganan atau tidak *churn*, yaitu dengan memberikan insentif retensi terhadap pelanggan. Insentif retensi yang dimaksud adalah dengan memberikan suku bunga yang menarik, memberikan paket layanan yang menarik, memberikan prioritas pelayanan dan lain-lain dalam upaya untuk mempertahankan para customer atau pelanggan perusahaan telekomunikasi tersebut. Namun, kebijakan pemberian insentif retensi belum sepenuhnya dilakukan secara efektif. Karena jika insentif retensi tersebut diberikan secara merata kepada seluruh pelanggan, maka pengeluaran biaya tersebut menjadi tidak efektif dan dapat mengurangi keuntungan apabila pelanggan tersebut memang loyal dan tidak ingin berhenti menggunakan layanan.

**Goals :**
Berdasarkan permasalahan di atas, perusahaan ingin memiliki kemampuan untuk memprediksi kemungkinan seorang pelanggan akan berhenti menggunakan layanan *(churn)* atau tidak, sehingga perusahaan dapat memfokuskan upaya untuk mempertahankan para pelanggannya pada pelanggan yang terindikasi untuk *churn*. 

Selain itu, perusahaan ingin mengetahui berbagai macam faktor-faktor yang mempengaruhi pelanggan bertahan, sehingga mereka dapat membuat program kebijakan yang tepat sasaran untuk mengurangi jumlah pelanggan yang berhenti berlangganan *(churn)*.

**Analytic Approach :**
Jadi yang akan dilakukan adalah menganalisis data untuk menemukan pola yang membedakan pelanggan yang akan berhenti menggunakan layanan *(churn)* atau tidak.

Kemudian akan membangun permodelingan machine learning klasifikasi yang akan membantu perusahaan untuk dapat memprediksi seorang pelanggan akan berhenti menggunakan layanan *(churn)* atau tidak.

**Metrix Evaluation :**
Error 1 : False Positive (customer yang aktualnya tidak churn,diprediksi churn) Konsekuensi biaya Sebesar $63

Error 2 : False Negative (customer yang aktualnya churn,diprediksi tidak akan churn) Konsekuensi: kehilangan pelanggan dan kehilangan biaya sebesar $315

Berdasarkan konsekuensinya, maka sebisa mungkin yang akan kita lakukan adalah membuat model yang dapat mengurangi customer churn dari perusahaan tersebut, khususnya jumlah False Negative (customer yang aktualnya churn tetapi diprediksi tidak akan churn), tetapi juga dapat meminimalisir pemberian insentif yang tidak tepat. Jadi nanti metric utama yang akan kita gunakan adalah f2_score, karena recall kita anggap dua kali lebih penting daripada precision. Kehilangan customer memakan biaya sebesar $315 sedangkan mempertahankan customer memakan biaya yang jauh lebih sedikit https://blog.usetada.com/id/retensi-lebih-menguntungkan-daripada-akuisisi#

## **Data Understanding**
- Sumber data : https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Pembuat data : IBM
- Periode pembuatan data : Juli 2019

### Attribute Information
Dataset ini berisi informasi tentang :

- Informasi demografi pelanggan yaitu Dependents.
- Service yang digunakan oleh pelanggan : Online Security, Online Backup, Internet Service, Device Protection, Tech Support
- Informasi akun pelanggan : tenure, Contract, PaperlessBilling, dan MonthlyCharges
- Pelanggan yang berhenti berlangganan – kolomnya disebut Churn

| Attribute | Data Type,Length	| Description |
| --- | --- | --- |
| Dependents | Text | Whether the customer has dependents or not. |
| tenure | Integer | Number of months the customer has stayed with the company |
| OnlineSecurity | Text | Indicates if the customer subscribes to an additional online security service provided by the company. |
| OnlineBackup | Text | Indicates if the customer subscribes to an additional online backup service provided by the company.Gender of candidate |
| InternetService | Text | Indicates if the customer subscribes to Internet service with the company. |
| DeviceProtection | Text | Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company |
| TechSupport | Text | Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times. |
| Contract | Text | Indicates the customer’s current contract type. |
| PaperlessBilling | Text | Indicates if the customer has chosen paperless billing. |
| MonthlyCharges | Float | Indicates the customer’s current total monthly charge for all their services from the company. |
| Churn (Target) | Text | Whether the customer churns or not.( Yes = the customer left the company this quarter. No = the customer remained with the company.) |

### **Cost Benefit Analysis**
**Cost Benefit Analysis tanpa penerapan Machine Learning :** 

- Biaya retensi Cutomer : $63
- Biaya kehilangan Cutomer : $315

Pada situasi tanpa menggunakan machine learning, perusahaan tidak dapat memprediksi Cutomer yang akan churn, sehingga retensi diberikan kepada seluruh Cutomer.

- Jumlah Cutomer yang diberi program retensi: 971 Cutomer
- Jumlah Cutomer yang sebenarnya churn: 250 Cutomer

Estimasi biaya :

- (971 * $63) + (258 * $315) = 61173 + 81270 = $142,443

Maka biaya yang harus dikeluarkan perusahaan setiap bulannya adalah $140,523 untuk 971 pelanggan.

**Cost Benefit Analysis dengan penerapan Machine Learning :**

Dengan menggunakan machine learning, program retensi hanya ditawarkan kepada pelanggan yang diprediksi akan churn.

- Jumlah pelanggan yang diberi program retensi: 564 pelanggan
- Jumlah pelanggan yang sebenarnya churn: 250 pelanggan

Estimasi biaya :

- (564 * $63) + (258 * $315) = 35532 + 81270 = $116,802

Maka Biaya yang harus dikeluarkan perusahaan untuk setiap bulannya $116,802 untuk 971 customer 

**Perbandingan menggunakan Machine Learning dan Tidak Menggunakan Machine Learning**

- Estimasi biaya tanpa machine learning : $142443
- Estimasi biaya dengan machine learning : $116,802

**Persentase Penurunan**

- Penurunan potensi kerugian: $140,523 - $116,802 = $26,241

Persentase Penurunan : (25641/142443) * 100% = 18%

Berdasarkan perhitungan biaya estimasi, penerapan machine learning dapat mengurangi potensi kerugian perusahaan hingga 18.7%. Ini menunjukkan efisiensi dalam alokasi sumber daya untuk program retensi, dengan menargetkan secara lebih tepat pelanggan yang berpotensi untuk churn.

### **Model Limitation**
Perlu diperhatikan bahwa permodelingan ini hanya berlaku untuk :

- Model ini hanya dapat diandalkan untuk data dengan rentang `tenure` antara 0 hingga 72 bulan.
- Analisis dan prediksi model tidak berlaku untuk data dengan `MonthlyCharges` kurang dari 18.8 atau lebih besar dari 118.65.
- Jenis `Contract` yang dapat diproses oleh model terbatas pada 'Month-to-month', 'One year', dan 'Two Year'.
- Model hanya relevan untuk kasus dengan `InternetService` yang dapat diakomodasi sebagai 'DSL', 'Fiber Optic', dan 'No'.
- Variabel `Dependent` dan `Paperless Billing` harus memiliki nilai 'Yes' atau 'No' agar hasil prediksi model berlaku.
- Fitur-fitur seperti `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, dan `TechSupport` hanya dapat memproses nilai 'Yes', 'No', atau 'No internet service',  dan nilai lainnya tidak valid.

### **Conclusion**
Setelah melakukan permodelingan dan mendapatkan best model dapat disimpulkan bahwa : 
- Metric yang digunakan dalam pemodelan adalah F2_score, dengan penekanan pada recall yang dianggap dua kali lebih penting daripada precision.

- Berdasarkan Hyperparameter Tuning untuk best model kita yaitu CatBoost Classifier adalah : 
    - `bagging_temperature` : 0.8817202123338397,
    - `depth` : 3,
    - `iterations` : 53 
    - `l2_leaf_reg` : 6.614442749813741,
    - `learning_rate` : 0.15975765150802665 , 
    - `scale_pos_weight` : 3,
    - `subsample`: 0.5859265495238215  
    <br>
- F2_score terbaik yang dicapai adalah 0.732097.
<br>
- Features/kolom yang paling berpengaruh pada model CatBoost Classifier, berdasarkan tingkat kepentingannya, adalah:
    - `contract` dengan poin sebesar 39.185476
    - `tenure` dengan poin sebesar 26.353524
    - `InternetService` dengan poin sebesar 18.203443
    - `MonthlyCharges` dengan poin sebesar 7.228902
    - `PaperlessBilling` dengan poin sebesar 3.490441
    - `TechSupport` dengan poin sebesar 2.265196
    - `Dependents` dengan poin sebesar 1.339233
    - `OnlineSecurity` dengan poin sebesar 0.989033
    - `OnlineBackup` dengan poin sebesar 0.903804
    - `DeviceProtection` dengan poin sebesar 0.040949.
<br><br>
- Berdasarkan hasil classification report, pemodelan menggunakan model CatBoost dengan metric f2_score :
    - Score Precision : 40.6%
    - Score Recall : 91.6%
    - Score Accuracy : 73.2%
<br><br>
-  Berdasarkan Cost Benefit Analysis : 
    - Penerapan model Machine Learning menghasilkan penghematan biaya sebesar $26,241 atau 18% dibandingkan dengan skenario tanpa Machine Learning.
    - Tanpa Machine Learning, biaya yang dikeluarkan perusahaan mencapai $142,443 untuk 971 pelanggan.
    - Dengan penerapan Machine Learning, biaya dapat ditekan menjadi $116,802 untuk jumlah pelanggan yang sama.

### **Recommendations**
Rekomendasi aksi yang dapat dilakukan perusahaan untuk meminimalisir jumlah pelanggan yang akan *chur*  :

1. Promosi Migrasi Kontrak:
    - Tawarkan insentif menarik kepada pelanggan dengan `contract` jangka pendek (Month-to-month) untuk beralih ke `contract` jangka panjang (One year atau Two year). Hal ini dapat mengurangi potensi *churn* karena pelanggan cenderung lebih loyal pada contract jangka panjang.
<br><br>
2. Program Loyalty Berbasis `tenure`:
    - Implementasikan Customer Loyalty Program yang memberikan reward proporsional dengan masa tenure pelanggan. Semakin lama pelanggan bertahan, semakin besar reward yang diperoleh. Strategi ini mendorong pelanggan untuk memperpanjang masa tenure mereka.
    <br><br>
3. Diskon Khusus untuk Pelanggan Berpotensi *Churn*:
    - Berikan diskon atau potongan harga pada `MonthlyCharges` untuk pelanggan yang memiliki indikasi atau prediksi churn, terutama bagi mereka dengan MonthlyCharges yang tinggi. Hal ini dapat menjadi insentif efektif untuk mempertahankan pelanggan yang berpotensi beralih ke perusahaan layanan lain.
<br><br>
4. Penawaran Khusus Internet Fiber Optic:
    - Sediakan layanan Internet Fiber Optic dengan harga yang lebih kompetitif. Berdasarkan analisis, rata-rata MonthlyCharges untuk Fiber Optic lebih tinggi.
    - Penawaran harga yang lebih terjangkau dapat mengurangi kemungkinan churn, terutama bagi pelanggan yang cenderung meninggalkan layanan akibat biaya yang tinggi.

Dengan mengimplementasikan langkah-langkah ini, perusahaan dapat secara proaktif meminimalisir potensi churn dan membangun hubungan yang lebih kuat dengan pelanggan, melalui kombinasi insentif, program loyalitas, dan penyesuaian layanan.


Hal-hal yang bisa dilakukan untuk mengembangkan project dan modelnya lebih baik lagi diantaranya:

1. Penambahan Fitur Sentimen Pelanggan:
    - Mengintegrasikan analisis sentimen pada feedback atau interaksi pelanggan dapat memberikan wawasan mengenai perasaan mereka terhadap layanan. Hal ini dapat menjadi indikator potensial untuk churn jika terdapat pola sentimen negatif yang konsisten.
<br><br>
2. Pemodelan Interaksi Produk:
    - Menambahkan fitur yang merepresentasikan interaksi antar produk atau layanan dapat membantu dalam memahami bagaimana kombinasi penggunaan berbagai produk dapat memengaruhi keputusan pelanggan. Model dapat menjadi lebih presisi dalam mengidentifikasi pola-pola perilaku yang berkaitan dengan churn.
<br><br>
3. Penyempurnaan Data Churn:
    - Melakukan penyempurnaan pada data kelas minoritas (Churn) dengan mengumpulkan informasi lebih lanjut atau mencari sumber data tambahan dapat meningkatkan keakuratan model. Pengembangan ini penting untuk menghadapi perubahan perilaku pelanggan.
<br><br>
4. Ensemble Model:
    - Mengimplementasikan ensemble model dengan menggabungkan hasil dari beberapa algoritma machine learning dapat meningkatkan keandalan prediksi. Pendekatan ini dapat membantu model untuk lebih tangguh terhadap variasi data dan kompleksitas pola.
<br><br>
    
Dengan menyertakan elemen-elemen ini dalam pengembangan model, diharapkan perusahaan dapat mencapai tingkat akurasi dan keandalan yang lebih tinggi dalam mengidentifikasi pelanggan yang berpotensi untuk churn.
