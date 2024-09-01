import gradio as gr
import pickle
import numpy as np
import pandas as pd

classes = ['Apple','Banana', 'Blackgram', 'ChickPea', 'Coconut', 'Coffee', 'Cotton', 'Grapes',
           'Jute', 'KidneyBeans', 'Lentil', 'Maize', 'Mango', 'MothBeans', 'MungBean', 'Muskmelon',
           'Orange', 'Papaya', 'PigeonPeas', 'Pomegranate', 'Rice', 'Watermelon']

# load model and scaler
loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))
model_files = {
    "Logistic Regression": "model_lr.pkl",
    "KNN": "model_knn.pkl",
    "Support Vector": "model_SVC.pkl",
    "Desicion Tree": "model_tree.pkl",
    "Random Forest": "model_rf.pkl",
    "AdaBoost": "model_AB_boost.pkl",
    "Bagging": "model_bagging_DT.pkl"
}
#result = loaded_model.score(X_test, Y_test)

def prediction(Nitrogen,Phosphorus,Potassium,Temperature,Humidity,pH_Value,Rainfall, model):
    # diccionary of the new input
    dic = {}
    values = [Nitrogen,Phosphorus,Potassium,Temperature,Humidity,pH_Value,Rainfall]
    for feature,value in zip(loaded_scaler.feature_names_in_, values):
        dic[feature] = [value]
    # scaler transform
    new_input = loaded_scaler.transform( pd.DataFrame(data=dic))
    # model prediction
    loaded_model = pickle.load(open(model_files[model], 'rb'))
    prediction_proba = loaded_model.predict_proba(new_input)[0]

    return {classes[i]: prediction_proba[i] for i in range(len(classes))}

demo = gr.Interface(
    fn=prediction,
    inputs=[gr.Slider(0, 140, value=0, label="Nitrogen"),
            gr.Slider(0, 145, value=0, label="Phosphorus"),
            gr.Slider(0, 200, value=0, label="Potassium"),
            gr.Slider(0, 50, value=0, label="Temperature"),
            gr.Slider(0, 100, value=0, label="Humidity"),
            gr.Slider(0, 10, value=0, label="pH_Value"),
            gr.Slider(0, 300, value=0, label="Rainfall"),
            gr.Dropdown(["Logistic Regression",
                         "KNN", "Support Vector",
                         "Desicion Tree","Random Forest",
                         "AdaBoost", "Bagging"
                         ], 
                         label="Models", 
                         info="Select the model!")
            ],
    outputs=gr.Label(num_top_classes=5),
    title="Crop recommendation"
)

demo.launch(share=True)
