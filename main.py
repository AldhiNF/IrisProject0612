import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(layout="centered", page_title="Iris Classification", page_icon=":cherry_blossom:")

model = joblib.load("model.joblib")

def get_predictions (data:pd.DataFrame, model):
    """Get Predictions

    Args:
        data (pd.DataFrame): dataFrame
        model (_type_): model classifier
    """

    predictions = model.predict(data)
    predict_proba = model.predict_proba(data)

    map_label = {
        0: "Iris-setosa",
        1: "Iris-versicolor",
        2: "Iris-virginica"
    }

    prediction_label = map(lambda x: map_label[x], list(predictions))

    return{
        "predictions": predictions,
        "prediction_label": list(prediction_label),
        "predict_proba": predict_proba
    }

st.title("Iris Classification", width="stretch")
st.write("Get your iris species!")

left, right = st.columns(2, gap="medium")

left.subheader("Sepal Information")
sepal_lenght = left.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=0.0)
sepal_width = left.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=0.0)

right.subheader("Petal Information")
petal_lenght = right.number_input("Petal Length", min_value=0.0, max_value=10.0, value=0.0)
petal_width = right.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.0)

predict = st.button("Predict", use_container_width=True)

if predict:
    df = pd.DataFrame(
        [[sepal_lenght, sepal_width, petal_lenght, petal_width]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )
    st.write(df)

    # Predict
    result = get_predictions(df, model)

    label = result["prediction_label"][0]
    prediction = result["predictions"][0]
    prob = result["predict_proba"][0]

    st.write(f"Your Iris Species is **{label}** ({prob})")