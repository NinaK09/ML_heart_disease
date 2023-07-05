import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from sklearn.naive_bayes import GaussianNB
import pickle


def prepare_data():
    data_file = pd.read_csv('heart_train-test.csv', sep=',')  # wczytanie pliku
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


X, y = prepare_data()
model = GaussianNB()

#Standaryzacja
scaler = StandardScaler()
X = scaler.fit_transform(X)
model.fit(X,y)


#zapis wytrenowanego modelu do pliku
file = "GaussianNB_Trained"
file_model = open(file, "wb")
pickle.dump(model, file_model)
file_model.close()
print("pickled!")