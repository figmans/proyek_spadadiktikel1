#import library streamlit
import streamlit as st

# import library pandas
import pandas as pd

# import library sklearn
import sklearn

#import library sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import re
import string

#Panggil file Fake.csv dan True.csv lalu simpan dalam dataframe
berita_palsu = pd.read_csv("Fake.csv")
berita_benar = pd.read_csv("True.csv")

berita_palsu["class"] = 0
berita_benar["class"] = 1

# Input shape
# berita_palsu.shape, berita_benar.shape

# Pengumpulan Data
berita_palsu_tes_manual = berita_palsu.tail(10)
for i in range(23480,23470,-1):
    berita_palsu.drop([i], axis=0, inplace=True)
berita_benar_tes_manual = berita_benar.tail(10)
for i in range(21416,21406,-1):
    berita_benar.drop([i], axis=0, inplace=True)

# Tes Manual Berita
berita_tes_manual = pd.concat([berita_palsu_tes_manual, berita_benar_tes_manual], axis=0)
berita_tes_manual.to_csv("Tes_Manual.csv")

# Penggabungan Berita
berita_merge = pd.concat([berita_palsu, berita_benar], axis=0)

# Untuk mengtahui Kolom
# berita_merge.columns

# Menghapus kolom title, subjek dan date
berita = berita_merge.drop(["title", "subject","date"], axis = 1)


# mengecek berita yang kosong
# berita.isnull().sum()

#Mengacak Data
berita = berita.sample(frac = 1)

berita.reset_index(inplace = True)
berita.drop(["index"], axis = 1, inplace = True)
# berita.columns

# Membuat fungsi untuk merapihkan tulisan yang berisi link, huruf besar dan lain lain
def wordopt(name):
    text = name.lower()
    text = re.sub('\[.*?\]', '', name)
    text = re.sub("\\W"," ",name) 
    text = re.sub('https?://\S+|www\.\S+', '', name)
    text = re.sub('<.*?>+', '', name)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', name)
    text = re.sub('\n', '', name)
    text = re.sub('\w*\d\w*', '', name)    
    return text

berita["text"] = berita["text"].apply(wordopt)

#Membuat variabel X dan Variabel Y 
x = berita["text"]
y = berita["class"]

#Memisahkan dataset menjadi dataset menjadi training set dan testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

#Membuat fungsi merubah teks menjadi vektor
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

#Melakukan Analisis Tahap Pertama Yang Logistic Regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xv_train,y_train)
pred_lr=LR.predict(xv_test)
LR.score(xv_test, y_test)

#Melakukan Analisis Tahap Kedua Yaitu Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
DT.score(xv_test, y_test)


#Melakukan Analisis Tahap Ketiga Yaitu Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)
pred_gbc = GBC.predict(xv_test)
GBC.score(xv_test, y_test)


#Melakukan Analisis Tahap Keempat Yaitu Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)
pred_rfc = RFC.predict(xv_test)
RFC.score(xv_test, y_test)

#Model Testing Dengean Memasukkan Data Manual
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

    return st.text("\n\nPrediksi LR: {} \nPrediksi DT: {} \nPrediksi GBC: {} \nPrediksi RFC: {}".format(output_lable(pred_LR[0]), 
                                                                                                              output_lable(pred_DT[0]), 
                                                                                                              output_lable(pred_GBC[0]), 
                                                                                                              output_lable(pred_RFC[0])))
st.title("Pengecek Berita Palsu dan Asli")

with st.form(key = "form1"):
    name = st.text_input(label = "Silahkan Masukkan Berita")
    submit  = st.form_submit_button(label = "Submit this form")
st.text(manual_testing(name))
