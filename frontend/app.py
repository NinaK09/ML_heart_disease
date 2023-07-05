from flask import Flask, render_template, request, redirect
import requests
import json
import plotly.graph_objects as go
#=============

app = Flask("app")
#flask run
#port 5000

@app.route("/", methods = ['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        user_input = request.form.to_dict() #pobranie danych z formularza
        float_keys = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Asthma', 'KidneyDisease',
                      'SkinCancer','PhysicalHealth', 'MentalHealth', 'PhysicalActivity', 'SleepTime']
        for elm in float_keys:
            user_input[elm] = float(user_input[elm])

        payload = {"input_json_form": user_input}
        r = requests.post(url="http://backend_inz:8000/check-health-from-form", data = json.dumps(payload))
        if(len(r.json()) == 3):
            result = bool(r.json()[1])
        else:
            result = "error"
        return render_template('form-final.html', heart_result=result)
    else:
        return render_template('form-final.html',  heart_result = "")


@app.route("/show-results")
def clf_results():
    r = requests.get(url="http://backend_inz:8000/clf-results")
    results = r.json()
    cols_names = ["model", "confusion_matrix", "recall", "precision", "accuracy"]

    for model_result in results:
        confusion_matrix_dict = model_result['confusion_matrix'] #słownik z wynikami macierzy
        #odpowiednia kolejność elm. macierzy pomyłek dla poprawnego oznaczenia fragmentow mapy cieplnej
        conf_matrix = [[confusion_matrix_dict['tn'], confusion_matrix_dict['fp']],[confusion_matrix_dict['fn'], confusion_matrix_dict['tp']]]
        #tworzenie mapy ciepła
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=["Negative", "Positive"],
            y=["Positive", "Negative"],
            text=[["TN = " + str(confusion_matrix_dict['tn']), "FP = " + str(confusion_matrix_dict['fp'])],
                  ["FN = " + str(confusion_matrix_dict['fn']), "TP = " + str(confusion_matrix_dict['tp'])]],
            texttemplate="%{text}",
            textfont={"size": 20}),
            layout={
                "title": model_result['model'],
                "xaxis": {"title": "Predicted value"},
                "yaxis": {"title": "Real value"}}
        )

        #zapisanie obrazka do folderu static/conf_matrix
        img_name = model_result['model'] + ".jpg"
        path = "static/conf_matrix/" + img_name
        fig.write_image(path)
        model_result["image name"] = img_name

    #FN malejąco
    results_lowest_FN = sorted(results, key=lambda j: j["confusion_matrix"]["fn"])
    #pełność rosnąco
    results_sorted = sorted(results_lowest_FN, key = lambda i: i['recall'], reverse=True)


    return render_template('modelsComparison.html', results_keys = cols_names, models = results_sorted)
