"""
match_embedding_playground.py — takim embedding'i ile mac sonucu tahmini
=============================================================================
Kaggle notebook'una hucre hucre yapistirmak icin dusunuldu. Amac: takim
isimlerini (kategorik veri) bir Embedding katmanina sokup, modelin
"hangi takim hangi takime nasil benziyor/farkli" bilgisini KENDI
KENDINE, sayisal bir vektor olarak ogrenmesini gormek.

Beklenen veri seti kolonlari (Kaggle'daki tipik "international football
results" tarzi setler genelde boyle): home_team, away_team, home_score,
away_score (ve genelde date, tournament gibi ekstra kolonlar da olur,
bu iskelette kullanilmiyor ama istersen ekleyebilirsin).
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# --- 1) VERI YUKLEME ---
# Kaggle'da: dosya path'i genelde /kaggle/input/<dataset-adi>/<dosya>.csv
# seklinde olur, notebook'un sag panelinde "Add Input" ile veri setini
# eklersen path'i otomatik gorursun.
df = pd.read_csv("/kaggle/input/DEGISTIR-BUNU/results.csv")
print(df.head())
print(df.shape)

# --- 2) SONUCU 3 SINIFA CEVIRME (ev sahibi kazanir / berabere / deplasman kazanir) ---
def match_result(row):
    if row["home_score"] > row["away_score"]:
        return 0  # ev sahibi kazanir
    elif row["home_score"] == row["away_score"]:
        return 1  # beraberlik
    else:
        return 2  # deplasman kazanir

df["result"] = df.apply(match_result, axis=1)

# --- 3) TAKIM ISIMLERINI SAYISAL ID'YE CEVIRME (embedding'in girdisi bu) ---
# Embedding katmani, takim ISMINI degil, takim ID'sini (0, 1, 2, ...)
# alir -- bu yuzden her benzersiz takima bir tam sayi atiyoruz.
all_teams = pd.concat([df["home_team"], df["away_team"]]).unique()
team_to_id = {team: i for i, team in enumerate(all_teams)}
num_teams = len(all_teams)
print(f"Toplam benzersiz takim sayisi: {num_teams}")

df["home_team_id"] = df["home_team"].map(team_to_id)
df["away_team_id"] = df["away_team"].map(team_to_id)

# --- 4) TRAIN/TEST AYIRMA ---
# Basit bir rastgele ayirma ile basla; istersen sonra TARIH bazli
# ayirmayi dene (gecmis maclarla egit, en yakin tarihli maclari test
# et) -- gercek dunyada daha anlamli bir degerlendirme olur.
from sklearn.model_selection import train_test_split

X = df[["home_team_id", "away_team_id"]].values
y = df["result"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


# --- 5) MODEL -- embedding'in gercek yeri burasi ---
EMBEDDING_DIM = 8  # --- DENE / DEGISTIR: 4, 16, 32 ile de dene, fark nasil degisiyor gor

home_input = layers.Input(shape=(1,), name="home_team_id")
away_input = layers.Input(shape=(1,), name="away_team_id")

# AYNI embedding katmanini hem ev sahibi hem deplasman icin kullaniyoruz
# (bir takimin kendi ozellikleri, ev/deplasman fark etmeksizin ayni
# olmali mantiken) -- bunun icin katmani BIR KERE tanimlayip ikisine de
# uyguluyoruz.
team_embedding = layers.Embedding(input_dim=num_teams, output_dim=EMBEDDING_DIM, name="team_embedding")

home_vec = layers.Flatten()(team_embedding(home_input))
away_vec = layers.Flatten()(team_embedding(away_input))

combined = layers.Concatenate()([home_vec, away_vec])
x = layers.Dense(32, activation="relu")(combined)
x = layers.Dropout(0.3)(x)
x = layers.Dense(16, activation="relu")(x)

# 3 sinif: ev sahibi kazanir / berabere / deplasman kazanir
outputs = layers.Dense(3, activation="softmax", name="result_probs")(x)

model = models.Model(inputs=[home_input, away_input], outputs=outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# --- 6) EGITIM ---
history = model.fit(
    {"home_team_id": X_train[:, 0], "away_team_id": X_train[:, 1]}, y_train,
    validation_data=({"home_team_id": X_test[:, 0], "away_team_id": X_test[:, 1]}, y_test),
    epochs=20, batch_size=64,
)

# --- 7) OGRENILEN TAKIM VEKTORLERINI GORSELLESTIRME ---
# Bu kisim en eglenceli olan -- egitim bitince, model her takim icin
# kendiliginden bir vektor uretmis oluyor, bunlari 2 boyuta indirip
# (PCA ile) bir grafikte "hangi takimlar birbirine yakin" diye
# gorebilirsin.
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

learned_weights = team_embedding.get_weights()[0]  # shape: (num_teams, EMBEDDING_DIM)
pca = PCA(n_components=2)
coords_2d = pca.fit_transform(learned_weights)

plt.figure(figsize=(14, 10))
plt.scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.3)
# --- DENE / DEGISTIR: hepsini degil, sadece bildigin/ilgini ceken
# birkac takimin ismini grafikte etiketle, hepsini yazarsan okunmaz
sample_teams = list(team_to_id.keys())[:30]
for team in sample_teams:
    i = team_to_id[team]
    plt.annotate(team, (coords_2d[i, 0], coords_2d[i, 1]), fontsize=8)
plt.title("Modelin kendiliginden ogrendigi takim vektorleri (2B'ye indirgenmis)")
plt.show()
