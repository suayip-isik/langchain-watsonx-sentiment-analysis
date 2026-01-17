# LangChain IBM Watsonx Sentiment Analysis Project

Bu proje, IBM Watsonx (Granite) modelini kullanarak LangChain ile metin analizi ve duygu analizi (sentiment analysis) yapan bir Python uygulamasıdır. Proje, müşteri geri bildirimlerini, yorumları veya herhangi bir metni analiz ederek özet ve duygu analizi çıkarır.

## 📋 İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Özellikler](#özellikler)
- [Kullanım Senaryoları](#kullanım-senaryoları)
- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Yapılandırma](#yapılandırma)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Mimari Açıklama](#mimari-açıklama)
- [Örnekler](#örnekler)
- [Sorun Giderme](#sorun-giderme)

## 🎯 Proje Hakkında

Bu proje, IBM Watsonx'in Granite-4-H-Small modelini kullanarak üç aşamalı bir metin analizi pipeline'ı oluşturur:

1. **Anahtar Kelime Çıkarma (Keyword Extraction)**: Metinden en önemli anahtar kelimeleri çıkarır
2. **Duygu Özeti (Sentiment Summary)**: Anahtar kelimelerden yola çıkarak metnin genel duygusunu özetler
3. **Rafine Etme (Refinement)**: Özeti daha kısa ve kesin hale getirir

Proje, LangChain'in modern **LCEL (LangChain Expression Language)** yaklaşımını kullanarak bu işlemleri sıralı bir şekilde gerçekleştirir. Deprecated `LLMChain` ve `SequentialChain` yerine `RunnableLambda` ve pipe operatörü (`|`) kullanılmaktadır.

## ✨ Özellikler

- 🔗 **IBM Watsonx Entegrasyonu**: IBM'in güncel Granite-4-H-Small modelini kullanır
- 🔄 **Modern LCEL Yapısı**: LangChain Expression Language ile üç aşamalı sıralı işlem akışı
- 🔐 **Güvenli Yapılandırma**: `.env` dosyası ile güvenli credential yönetimi
- 📊 **Sentiment Analysis**: Metinlerin duygusal tonunu analiz eder
- 🎯 **Keyword Extraction**: Metinlerden önemli anahtar kelimeler çıkarır
- ✨ **Text Refinement**: Çıktıları daha okunabilir hale getirir
- ⚡ **Deprecated-Free**: Modern LangChain API'leri kullanır, uyarı vermez

## 🚀 Kullanım Senaryoları

Bu proje aşağıdaki durumlarda kullanılabilir:

### 1. Müşteri Geri Bildirim Analizi
- Müşteri yorumlarını ve şikayetlerini analiz etme
- Ürün/hizmet geri bildirimlerini kategorize etme
- Müşteri memnuniyet seviyesini ölçme

### 2. Sosyal Medya Analizi
- Sosyal medya gönderilerinin duygu analizi
- Marka itibarı takibi
- Trend analizi

### 3. İçerik Analizi
- Blog yazıları, makaleler ve içeriklerin analizi
- Metin özetleme
- İçerik kategorizasyonu

### 4. Anket ve Araştırma
- Açık uçlu anket cevaplarının analizi
- Nitel veri analizi
- Araştırma verilerinin özetlenmesi

### 5. Destek Bileti Analizi
- Müşteri destek taleplerinin kategorize edilmesi
- Acil durumların tespit edilmesi
- Destek metriklerinin iyileştirilmesi

## 📦 Gereksinimler

### Python Versiyonu
- Python 3.8 veya üzeri

### Gerekli Paketler
- `langchain` (1.2.6+)
- `langchain-core` (1.2.7+)
- `langchain-ibm` (1.0.2+)
- `python-dotenv` (0.19.0+)

**Not**: Bu proje modern LangChain LCEL yaklaşımını kullandığı için `langchain-classic` paketine ihtiyaç duymaz.

### IBM Cloud Gereksinimleri
- IBM Cloud hesabı
- Watson Machine Learning servisi
- Watsonx API anahtarı
- Project ID

## 🔧 Kurulum

### 1. Projeyi Klonlayın veya İndirin

```bash
git clone <repository-url>
cd langchain-example
```

### 2. Sanal Ortam Oluşturun (Önerilen)

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

### 3. Gerekli Paketleri Kurun

```bash
pip install langchain langchain-core langchain-ibm python-dotenv
```

veya `requirements.txt` dosyası oluşturup:

```bash
pip install -r requirements.txt
```

**Not**: `langchain-classic` paketi artık gerekli değildir çünkü proje modern LCEL yaklaşımını kullanmaktadır.

## ⚙️ Yapılandırma

### 1. `.env` Dosyası Oluşturun

Proje kök dizininde `.env` dosyası oluşturun:

```env
WATSONX_URL=https://eu-de.ml.cloud.ibm.com
WATSONX_APIKEY=your_api_key_here
PROJECT_ID=your_project_id_here
```

### 2. IBM Cloud Yapılandırması

#### API Key Alma
1. IBM Cloud konsoluna giriş yapın
2. **Manage** > **Access (IAM)** > **API Keys** bölümüne gidin
3. Yeni bir API key oluşturun veya mevcut birini kullanın

#### Project ID Alma
1. IBM Cloud konsolunda Watson Machine Learning servisinize gidin
2. Proje detaylarından Project ID'yi kopyalayın

#### Bölge Seçimi
Türkiye için önerilen bölgeler:
- `eu-de` (Frankfurt) - Önerilen
- `eu-gb` (Londra) - Alternatif

Diğer bölgeler:
- `us-south` (Dallas)
- `us-east` (Washington)
- `jp-tok` (Tokyo)
- `au-syd` (Sydney)

## 🎮 Kullanım

### Temel Kullanım

```bash
python main.py
```

### Kod İçinde Kullanım

```python
# Kendi metninizi analiz etmek için
feedback_text = """
Your text here...
"""

result = workflow.invoke({"text": feedback_text})
print(result.get("refined_summary", result))
```

### Özelleştirme

#### Farklı Model Kullanma

```python
llm = WatsonxLLM(
    model_id="ibm/granite-4-h-small",  # Varsayılan model (güncel)
    # model_id="ibm/granite-3-8b-instruct",  # Eski model (deprecated)
    url=credentials["url"],
    apikey=credentials["apikey"],
    project_id=project_id,
    params={
        "max_new_tokens": 200  # Token sayısını artırın
    }
)
```

**Not**: `ibm/granite-3-8b-instruct` modeli deprecated durumdadır. Yeni projeler için `ibm/granite-4-h-small` kullanılması önerilir.

#### Prompt Şablonlarını Değiştirme

```python
keyword_prompt = PromptTemplate(
    input_variables=["text"],
    template="Your custom template here: {text}\n\nKeywords:"
)
```

## 📁 Proje Yapısı

```
langchain-example/
│
├── main.py              # Ana uygulama dosyası
├── main.ipynb           # Jupyter notebook versiyonu (opsiyonel)
├── .env                 # Ortam değişkenleri (oluşturulmalı)
├── .env.example         # Örnek .env dosyası (opsiyonel)
├── README.md            # Bu dosya
└── requirements.txt     # Python bağımlılıkları (opsiyonel)
```

## 🏗️ Mimari Açıklama

### İş Akışı (Workflow)

```
Input Text
    ↓
[Keyword Chain] → Keywords
    ↓
[Sentiment Chain] → Sentiment Summary
    ↓
[Refine Chain] → Refined Summary
    ↓
Output
```

### Bileşenler

1. **WatsonxLLM**: IBM Watsonx modelini sarmalayan LangChain LLM wrapper'ı
2. **PromptTemplate**: Her aşama için özelleştirilmiş prompt şablonları
3. **RunnableLambda**: Her aşamayı temsil eden fonksiyon tabanlı runnable bileşenleri
4. **LCEL Pipeline**: Pipe operatörü (`|`) ile birleştirilmiş sıralı workflow
5. **extract_text()**: LLM yanıtlarından metin içeriğini çıkaran yardımcı fonksiyon

### Modern Yaklaşım: LCEL (LangChain Expression Language)

Bu proje, deprecated `LLMChain` ve `SequentialChain` yerine modern LCEL yaklaşımını kullanır:

```python
# Eski yaklaşım (deprecated)
chain = LLMChain(llm=llm, prompt=prompt)
workflow = SequentialChain(chains=[chain1, chain2, chain3])

# Yeni yaklaşım (modern)
chain = prompt | llm
workflow = RunnableLambda(func1) | RunnableLambda(func2) | RunnableLambda(func3)
```

**Avantajları:**
- ✅ Deprecated uyarıları yok
- ✅ Daha esnek ve okunabilir kod
- ✅ LangChain'in gelecek versiyonlarıyla uyumlu
- ✅ Daha iyi performans

### Veri Akışı

```python
# Giriş
{"text": "I love this app but it crashes sometimes"}

# Adım 1: Keyword Extraction
→ {"text": "...", "keywords": "app, crashes, love"}

# Adım 2: Sentiment Summary
→ {"text": "...", "keywords": "...", "sentiment_summary": "Mixed feelings..."}

# Adım 3: Refinement
→ {"refined_summary": "Final concise summary"}
```

### Kod Yapısı Örneği

```python
# Her adım bir fonksiyon olarak tanımlanır
def extract_keywords(input_dict):
    result = (keyword_prompt | llm).invoke({"text": input_dict["text"]})
    keywords = extract_text(result)
    return {"text": input_dict["text"], "keywords": keywords}

# Fonksiyonlar RunnableLambda ile sarmalanır ve pipe ile birleştirilir
workflow = (
    RunnableLambda(extract_keywords)
    | RunnableLambda(generate_sentiment)
    | RunnableLambda(refine_summary)
)

# Kullanım
result = workflow.invoke({"text": "your text here"})
```

## 💡 Örnekler

### Örnek 1: Müşteri Geri Bildirimi

```python
feedback_text = """
I really enjoy the features of this app, but it crashes frequently, 
making it hard to use. The customer support is helpful, but response 
times are slow.
"""

result = workflow.invoke({"text": feedback_text})
refined_summary = result.get("refined_summary", result)
print(refined_summary)
# Çıktı: Duygu analizi ve özet
```

### Örnek 2: Ürün Yorumu

```python
review_text = """
This product exceeded my expectations! The quality is outstanding 
and the price is very reasonable. Highly recommend!
"""

result = workflow.invoke({"text": review_text})
print(result.get("refined_summary", result))
```

### Örnek 3: Şikayet Analizi

```python
complaint_text = """
I've been waiting for my order for over two weeks. The tracking 
information is not updated and customer service is not responding 
to my emails. Very disappointed.
"""

result = workflow.invoke({"text": complaint_text})
print(result.get("refined_summary", result))
```

## 🔍 Sorun Giderme

### Yaygın Hatalar ve Çözümleri

#### 1. ModuleNotFoundError: No module named 'dotenv'

**Çözüm:**
```bash
pip install python-dotenv
```

#### 2. Deprecated Uyarıları

**Çözüm:**
Bu proje modern LCEL yaklaşımını kullandığı için deprecated uyarıları görmezsiniz. Eğer eski kod tabanından geçiş yapıyorsanız:
- `LLMChain` yerine `RunnableLambda` kullanın
- `SequentialChain` yerine pipe operatörü (`|`) kullanın
- `.run()` yerine `.invoke()` kullanın

#### 3. API Key veya Project ID Hatası

**Çözüm:**
- `.env` dosyasının doğru konumda olduğundan emin olun
- Değişken isimlerinin doğru olduğunu kontrol edin
- IBM Cloud konsolundan API key ve Project ID'yi doğrulayın

#### 4. Bölge (Region) Hatası

**Çözüm:**
- `.env` dosyasındaki `WATSONX_URL` değerini kontrol edin
- Projenizin hangi bölgede oluşturulduğunu IBM Cloud konsolundan kontrol edin
- Bölge URL'lerinin doğru formatını kullanın: `https://{region}.ml.cloud.ibm.com`

#### 5. Model ID Hatası veya Deprecated Model Uyarısı

**Çözüm:**
- IBM Cloud konsolundan mevcut model ID'lerini kontrol edin
- Model ID formatı: `ibm/{model-name}`
- `ibm/granite-3-8b-instruct` deprecated durumdadır, `ibm/granite-4-h-small` kullanın
- Model lifecycle bilgileri için: https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-model-lifecycle.html

## 📚 Ek Kaynaklar

- [LangChain Dokümantasyonu](https://python.langchain.com/)
- [IBM Watsonx Dokümantasyonu](https://www.ibm.com/products/watsonx)
- [LangChain IBM Entegrasyonu](https://python.langchain.com/docs/integrations/llms/ibm_watsonx)

## 🤝 Katkıda Bulunma

1. Bu repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📝 Lisans

Bu proje örnek amaçlı oluşturulmuştur. Kullanımınız kendi sorumluluğunuzdadır.

## 👤 Yazar

Bu proje LangChain ve IBM Watsonx entegrasyonu için bir örnek uygulamadır.

## 🔄 Güncellemeler

- **v1.1.0**: Modern LCEL yaklaşımına geçiş, deprecated uyarıları giderildi
  - `LLMChain` ve `SequentialChain` yerine `RunnableLambda` kullanımı
  - Model güncellemesi: `ibm/granite-4-h-small`
  - `.invoke()` metodu kullanımı
- **v1.0.0**: İlk sürüm - Temel sentiment analysis pipeline'ı

---

**Not**: Bu proje, IBM Watsonx ve LangChain kullanarak metin analizi yapmak isteyen geliştiriciler için bir başlangıç noktasıdır. Özel ihtiyaçlarınıza göre özelleştirebilirsiniz.
