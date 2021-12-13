#!/usr/bin/env python
# coding: utf-8

# ### Deteksi Masalah Berita Palsu
# 
# Program ini dibuat dengan tujuan untuk dapat menyelesaikan permasalahan yang terjadi supaya tidak terjadi kekacauan pada lingkungan masyarakat saat ini. Dengan harapan apabila program ini dapat dibuat maka permasalahan yang terjadi di masyarakat mudah terselesaikan karena dapat dengan mudah menemukan berita palsu secara akurat melalui program ini. 
# 
# ### Kelompok 1 - UG05
# - Fachril Indra Gunawan [Universitas Gunadarma]
# - Didiek Trisatya [Universitas Dian Nuswantoro]
# - Andika Tri Kusuma [Universitas Gadjah Mada]
# 
# ### Memulai Pemrograman

# In[9]:


# import library pandas
import pandas as pd

# Import library numpy
import numpy as np

# Import library sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import re
import string


# In[10]:


#Panggil file Fake.csv dan True.csv lalu simpan dalam dataframe
berita_palsu = pd.read_csv("Fake.csv")
berita_benar = pd.read_csv("True.csv")


# In[11]:


# Tampilkan 10 baris awal dataset berita palsu dengan function head()
berita_palsu.head(10)


# In[12]:


# Tampilkan 10 baris awal dataset berita benar dengan function head()
berita_benar.head(10)


# In[13]:


berita_palsu["class"] = 0
berita_benar["class"] = 1


# In[14]:


# Input shape
berita_palsu.shape, berita_benar.shape


# In[15]:


# Pengumpulan Data
berita_palsu_tes_manual = berita_palsu.tail(10)
for i in range(23480,23470,-1):
    berita_palsu.drop([i], axis=0, inplace=True)
berita_benar_tes_manual = berita_benar.tail(10)
for i in range(21416,21406,-1):
    berita_benar.drop([i], axis=0, inplace=True)


# In[16]:


# Tes Manual Berita
berita_tes_manual = pd.concat([berita_palsu_tes_manual, berita_benar_tes_manual], axis=0)
berita_tes_manual.to_csv("Tes_Manual.csv")


# In[17]:


# Penggabungan Berita
berita_merge = pd.concat([berita_palsu, berita_benar], axis=0)
berita_merge.head(10)


# In[18]:


# Untuk mengtahui Kolom
berita_merge.columns


# In[19]:


# Menghapus kolom title, subjek dan date
berita = berita_merge.drop(["title", "subject","date"], axis = 1)


# In[20]:


# mengecek berita yang kosong
berita.isnull().sum()


# In[21]:


#Mengacak Data
berita = berita.sample(frac = 1)


# In[22]:


berita.head()


# In[23]:


berita.reset_index(inplace = True)
berita.drop(["index"], axis = 1, inplace = True)


# In[24]:


berita.columns


# In[25]:


berita.head()


# In[26]:


# Membuat fungsi untuk merapihkan tulisan yang berisi link, huruf besar dan lain lain
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


# In[27]:


berita["text"] = berita["text"].apply(wordopt)


# In[28]:


#Membuat variabel X dan Variabel Y 
x = berita["text"]
y = berita["class"]


# In[29]:


#Memisahkan dataset menjadi dataset menjadi training set dan testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# In[30]:


#Membuat fungsi merubah teks menjadi vektor
from sklearn.feature_extraction.text import TfidfVectorizer


# In[31]:


vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# # Tahap Pertama Logistic Regression

# In[32]:


from sklearn.linear_model import LogisticRegression


# In[33]:


LR = LogisticRegression()
LR.fit(xv_train,y_train)


# In[34]:


pred_lr=LR.predict(xv_test)


# In[35]:


LR.score(xv_test, y_test)


# In[36]:


print(classification_report(y_test, pred_lr))


# # # Tahap Kedua Decision Tree Classification

# In[37]:


from sklearn.tree import DecisionTreeClassifier


# In[38]:


DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# In[39]:


pred_dt = DT.predict(xv_test)


# In[40]:


DT.score(xv_test, y_test)


# In[41]:


print(classification_report(y_test, pred_dt))


# # # Tahap Ketiga Gradient Boosting Classifier

# In[42]:


from sklearn.ensemble import GradientBoostingClassifier


# In[43]:


GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)


# In[44]:


pred_gbc = GBC.predict(xv_test)


# In[45]:


GBC.score(xv_test, y_test)


# In[46]:


print(classification_report(y_test, pred_gbc))


# # # Tahap Keempat Random Forest Classifier

# In[47]:


from sklearn.ensemble import RandomForestClassifier


# In[48]:


RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)


# In[49]:


pred_rfc = RFC.predict(xv_test)


# In[50]:


RFC.score(xv_test, y_test)


# In[51]:


print(classification_report(y_test, pred_rfc))


# # # Model Testing Dengan Memasukkan Data Manual

# In[54]:


def output_lable(n):
    if n == 0:
        return "Berita Palsu"
    elif n == 1:
        return "Berita Asli"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    return print("\n\nPrediksi LR: {} \nPrediksi DT: {} \nPrediksi GBC: {} \nPrediksi RFC: {}".format(output_lable(pred_LR[0]), 
                                                                                                              output_lable(pred_DT[0]), 
                                                                                                              output_lable(pred_GBC[0]), 
                                                                                                              output_lable(pred_RFC[0])))


# In[55]:


news = str(input())
manual_testing(news)


# In[ ]:




