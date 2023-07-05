# Use of machine learning in heart disease risk prediction
First machine learning project, created for the easy and quick self diagnosis of heart disease risk for average human. The application itself is written using Python, both web application part and the resolve about which machine learning algorithm is the most suitable for this problem. User interface was written using Flask framework, backend was created with the help of FastAPI, and for the machine learning - Scikit-learn.

## Technologies
* Python
* Scikit-learn
* FastAPI
* Flask
* Docker

## Which machine learning model, and why?
To resolve, which algorithm is the best for deciding if user is prone to heart disease from their input, there were few metrics used:
* Confusion matrix and its elements;
* recall
* precision
* accuracy
Obviously, each metric is not equally important. The desired model should have the highest recall, lowest number of False Negative results in confusion matrix and highest accuracy, significance in that sequence. It all grants us an algorithm, that can properly classify a huge amount of people in risk of disease.
In this project, for this dataset, the best machine learning algorithm was ***Gaussian Naive Bayes***, that scored 78.76% recall, 71.43% accuracy and had only 1163 False Negatives for over 64 000 records.

## Dataset
[Personal Key Indicators of Heart Disease by Kamil Pytlak](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)
