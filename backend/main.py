from typing import Union
from fastapi import FastAPI
from typing import Dict
from pydantic import BaseModel
import numpy as np
import pickle
#=======================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
#========================================
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, LogisticRegressionCV, Perceptron
from sklearn.naive_bayes import  BernoulliNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
#=======================================

app = FastAPI()
#uvicorn main:app --reload

class FormData(BaseModel):
    input_json_form : Dict

@app.post("/check-health-from-form")
def check_health_from_form(data_sent: FormData):
    def load_model():
        #plik znajduje się w tym samym katalogu
        file_model = open("GaussianNB_Trained", "rb")
        model = pickle.load(file_model)
        file_model.close()
        return model

    def format_sent_data(data):
        #slownik z danymi; wartość klucza input_json_form
        form_data = data.input_json_form

        # przeksztalcenia typu 'AgeCategory': "AgeCategory_18-24" na "AgeCategory_18-24": 1.0 w słowniku
        form_data[form_data['AgeCategory']] = 1.0
        form_data.pop('AgeCategory')
        # to samo dla kolumn powstałych przez get_dummies()
        form_data[form_data['Sex']] = 1.0
        form_data.pop('Sex')
        form_data[form_data['Race']] = 1.0
        form_data.pop('Race')
        form_data[form_data['Diabetic']] = 1.0
        form_data.pop('Diabetic')
        form_data[form_data['GenHealth']] = 1.0
        form_data.pop('GenHealth')

        #kolumny które widzi model:
        data_columns_names = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking',
                              'PhysicalActivity', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer', 'Sex_Female', 'Sex_Male',
                              'AgeCategory_18-24', 'AgeCategory_25-29', 'AgeCategory_30-34', 'AgeCategory_35-39', 'AgeCategory_40-44',
                              'AgeCategory_45-49', 'AgeCategory_50-54', 'AgeCategory_55-59', 'AgeCategory_60-64', 'AgeCategory_65-69',
                              'AgeCategory_70-74', 'AgeCategory_75-79', 'AgeCategory_80 or older',
                              'Race_American Indian/Alaskan Native', 'Race_Asian', 'Race_Black', 'Race_Hispanic', 'Race_Other', 'Race_White',
                              'Diabetic_No', 'Diabetic_No, borderline diabetes', 'Diabetic_Yes', 'Diabetic_Yes (during pregnancy)',
                              'GenHealth_Excellent', 'GenHealth_Fair', 'GenHealth_Good', 'GenHealth_Poor', 'GenHealth_Very good']

        #wartości ze słownika form_data do listy numpy
        #np.array jest dwuwymiarowy, bo tylko takich używa model.predict()
        modified_data = np.array([[form_data['BMI'], form_data['Smoking'], form_data['AlcoholDrinking'], form_data['Stroke'], form_data['PhysicalHealth'],
                                  form_data['MentalHealth'], form_data['DiffWalking'], form_data['PhysicalActivity'],form_data['SleepTime'],
                                  form_data['Asthma'], form_data['KidneyDisease'], form_data['SkinCancer'],
                                  form_data.get('Sex_Female', 0.0), form_data.get('Sex_Male', 0.0),
                                  form_data.get('AgeCategory_18-24', 0.0), form_data.get('AgeCategory_25-29', 0.0), form_data.get('AgeCategory_30-34', 0.0),
                                  form_data.get('AgeCategory_35-39', 0.0), form_data.get('AgeCategory_40-44', 0.0), form_data.get('AgeCategory_45-49', 0.0),
                                  form_data.get('AgeCategory_50-54', 0.0), form_data.get('AgeCategory_55-59', 0.0), form_data.get('AgeCategory_60-64', 0.0),
                                  form_data.get('AgeCategory_65-69', 0.0), form_data.get('AgeCategory_70-74', 0.0), form_data.get('AgeCategory_75-79', 0.0),
                                  form_data.get('AgeCategory_80 or older', 0.0),form_data.get('Race_American Indian/Alaskan Native', 0.0),
                                  form_data.get('Race_Asian', 0.0), form_data.get('Race_Black', 0.0), form_data.get('Race_Hispanic', 0.0),
                                  form_data.get('Race_Other', 0.0), form_data.get('Race_White', 0.0),
                                  form_data.get('Diabetic_No', 0.0),form_data.get('Diabetic_No, borderline diabetes', 0.0),
                                  form_data.get('Diabetic_Yes', 0.0),form_data.get('Diabetic_Yes (during pregnancy)', 0.0),
                                  form_data.get('GenHealth_Excellent', 0.0), form_data.get('GenHealth_Fair', 0.0),form_data.get('GenHealth_Good', 0.0),
                                  form_data.get('GenHealth_Poor', 0.0), form_data.get('GenHealth_Very good', 0.0)
                                  ]]
                                 ,dtype=np.float64)

        #długość musi się zgadzać, by model mógł przewidzieć
        if(len(data_columns_names) == len(modified_data[0])):
            return modified_data
        else:
            return []


    def predict_from_form(data_nplist):
        model = load_model()
        y_pred = model.predict(data_nplist)
        return y_pred

    valid_data = format_sent_data(data_sent)
    if(len(valid_data) != 0):
        prediction = predict_from_form(valid_data)
        return str(prediction)
    else:
        return "Coś poszło nie tak. Spróbuj ponownie"


@app.get("/clf-results")
def models():
    def prepare_data():
        data_file = pd.read_csv('heart_train-test.csv', sep=',')
        # zamiana wartości kolumn na 0/1
        yes_no_columns = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'Asthma',
                          'KidneyDisease', 'SkinCancer']
        for i in yes_no_columns:
            data_file[i] = data_file[i].apply(lambda x: 1 if x == 'Yes' else 0)
        data_file['HeartDisease'] = data_file['HeartDisease'].apply(lambda x: 1 if x == 'Yes' else 0)
        # zmiana kategorycznych
        cat_columns = ['Sex','AgeCategory', 'Race', 'Diabetic', 'GenHealth']
        data_file = pd.get_dummies(data_file, columns=cat_columns)
        # zbiór wyjściowy
        y = data_file['HeartDisease'].values
        # zbiór wejściowy
        X = data_file.drop(columns=['HeartDisease']).values
        return X, y

    def make_model(X, y, clf):
        #podział zbiorów testowych (20%) i treningowych (80%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # standaryzacja
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        model = clf
        model.fit(X_train, y_train)
        # predykcja
        y_pred = model.predict(X_test)
        # macierz pomyłek, precyzja, dokładność, pełność
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        precision = precision_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_true=y_test, y_pred=y_pred)

        #z numpy.int64/numpy.float64 na int/float. nazwa modelu na str
        model_name = str(model)
        tn = tn.item()
        fp = fp.item()
        fn = fn.item()
        tp = tp.item()
        accuracy = accuracy.item()
        recall = recall.item()
        precision = precision.item()

        results = {"model": model_name,
                   "confusion_matrix": {
                       "tn": tn,
                       "fp": fp,
                       "fn": fn,
                       "tp": tp
                   },
                   "precision": precision,
                   "accuracy": accuracy,
                   "recall": recall
                   }

        return results

    model_result = []
    X, y = prepare_data()
    clfs = [kNN(),NearestCentroid(), Perceptron(), BernoulliNB(), LogisticRegressionCV(), ExtraTreeClassifier(), BaggingClassifier(), GradientBoostingClassifier(), SGDClassifier(), LinearDiscriminantAnalysis(), DecisionTreeClassifier(), ExtraTreesClassifier(), LogisticRegression(), RandomForestClassifier(), MLPClassifier(max_iter=500), AdaBoostClassifier(), GaussianNB()]
    for elm in clfs:
        try:
            res= make_model(X, y, elm)
            model_result.append(res)
        except Exception:
            print(Exception)
        else:
            pass

    return model_result
