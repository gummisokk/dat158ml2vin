import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from PIL import Image

red_wine = pd.read_csv("data/winequality-red.csv", sep=';')
white_wine = pd.read_csv("data/winequality-white.csv", sep=';')

stsmak, stred, stwhite = st.tabs(["Vinsmakaraen", "Rødvinsmodell info", "Kvitvinsmodell info"])


#red wine model

stred.title("Red wine model")

rwx = red_wine.drop('quality', axis = 1)
rwy = red_wine['quality']

rwx_train, rwx_test, rwy_train, rwy_test = train_test_split(rwx, rwy, test_size=0.25, random_state=42)

stred.write(f"split treningsset: {len(rwx_train)} testset: {len(rwx_test)}")

rwmodel = RandomForestClassifier(random_state=42)
rwmodel.fit(rwx_train, rwy_train)

rw_cvp = cross_val_predict(rwmodel, rwx_train, rwy_train, n_jobs=-1)
stred.write(f"cross val accuracy: {accuracy_score(rwy_train, rw_cvp)}")
stred.write("cross val forvirringsmatrise :")
stred.table(confusion_matrix(rwy_train, rw_cvp))

rwtest_pred = rwmodel.predict(rwx_test)
stred.write(f"test set accuracy: {accuracy_score(rwy_test, rwtest_pred)}")
stred.write("test set forvirringsmatrise :")
stred.table(confusion_matrix(rwy_test, rwtest_pred))


#white wine model

stwhite.title("White wine model")

wwx = white_wine.drop('quality', axis = 1)
wwy = white_wine['quality']

wwx_train, wwx_test, wwy_train, wwy_test = train_test_split(wwx, wwy, test_size=0.25, random_state=42)

stwhite.write(f"split treningsset: {len(wwx_train)} testset: {len(wwx_test)}")

wwmodel = RandomForestClassifier(random_state=42)
wwmodel.fit(wwx_train, wwy_train)

ww_cvp = cross_val_predict(wwmodel, wwx_train, wwy_train, n_jobs=-1)
stwhite.write(f"cross val accuracy:  {accuracy_score(wwy_train, ww_cvp)}")
stwhite.write("cross val forvirringsmatrise :")
stwhite.table(confusion_matrix(wwy_train, ww_cvp))

wwtest_pred = wwmodel.predict(wwx_test)
stwhite.write(f"test set accuracy: {accuracy_score(wwy_test, wwtest_pred)})")
stwhite.write("test set forvirringsmatrise :")
stwhite.table(confusion_matrix(wwy_test, wwtest_pred))



# "main gui"

stsmak.title("Vinsmakaren")

color = stsmak.segmented_control("Farge", ["Kvit", "Raud"])
fa = stsmak.slider("fixed acidity"       , 0.0, 10.0 , 7.0 , 0.1)
va = stsmak.slider("volatile acidity"    , 0.0, 1.0  , 0.3 , 0.01)
ca = stsmak.slider("citric acid"         , 0.0, 1.0  , 0.3 , 0.01)
rs = stsmak.slider("residual sugar"      , 0.0, 30.0 , 10.0, 0.1)
ch = stsmak.slider("chlorids"            , 0.0, 1.0  , 0.05, 0.001)
fs = stsmak.slider("free sulfur dioxide" , 0.0, 300.0, 40.0, 1.0)
ts = stsmak.slider("total sulfur dioxide", 0.0, 300.0, 80.0, 1.0)
de = stsmak.slider("density"             , 0.0, 2.0  , 1.0 , 0.0001)
ph = stsmak.slider("pH"                  , 0.0, 14.0 , 3.25, 0.01)
sp = stsmak.slider("sulphates"           , 0.0, 1.0  , 0.5 , 0.01)
al = stsmak.slider("alcohol"             , 0.0, 40.0 , 10.0, 0.1)

tomtglass = Image.open("kunst/tomtglass.png");
img = stsmak.image(tomtglass)

#gif
raudbildefil = open("kunst/raud.gif", "rb")
raud = base64.b64encode(raudbildefil.read()).decode("utf-8")
raudbildefil.close()
kvitbildefil = open("kunst/kvit.gif", "rb")
kvit = base64.b64encode(kvitbildefil.read()).decode("utf-8")
kvitbildefil.close()


if stsmak.button("smak på vinen"):
    if color == "Kvit":
        img.markdown(
            f'<img src="data:image/gif;base64,{kvit}">',
            unsafe_allow_html=True,
        )
        score = wwmodel.predict([[fa,va,ca,rs,ch,fs,ts,de,ph,sp,al]])
        stsmak.write(f"mmm, eg gir dinna ein {score[0]}")
    elif color == "Raud":
        img.markdown(
            f'<img src="data:image/gif;base64,{raud}">',
            unsafe_allow_html=True,
        )
        score = rwmodel.predict([[fa,va,ca,rs,ch,fs,ts,de,ph,sp,al]])
        stsmak.write(f"mmm, eg gir dinna ein {score[0]}")
    





#predictions = [cross_val_pred]
#classifier_names = ["Random Forest"]

#fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#for i, (prediction, classifier_name) in enumerate(zip(predictions, classifier_names)):
#    ConfusionMatrixDisplay.from_predictions(y_train, prediction, ax=axes[i])
#    axes[i].set_title(classifier_name)
#plt.show()