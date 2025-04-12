import streamlit as st
from streamlit_shap import st_shap
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import lime
from lime.lime_tabular import LimeTabularExplainer
import re
from fastapi import FastAPI, Request
import uvicorn

model = joblib.load('Model/GradientBoost_new.joblib')
scaler = joblib.load('Model/minmax_scaler_new.joblib')
X_train = joblib.load('Model/X_train_new.joblib')

if isinstance(X_train, np.ndarray):
    X_train = pd.DataFrame(X_train, columns=['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 
                                             'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate', 
                                             'exercise_angina', 'oldpeak', 'ST_slope'])

st.set_page_config(page_title="Heart Disease Prediction App", page_icon="❤️", layout="wide")

st.markdown("<h1 style='text-align: center;'>Heart Disease Prediction App ❤️</h1>", unsafe_allow_html=True)

def get_single_explanation(individual_shap_values, dataset):
    df = pd.DataFrame(individual_shap_values, index=dataset.columns, columns=["SHAP Value"])
    df["Absolute_Values"] = df["SHAP Value"].abs()
    df.sort_values(by="Absolute_Values", ascending=False, inplace=True)

    def apply_text_explanation(shap_value):
        if shap_value > 0:
            return f" has a positive impact, and <span style='color:green'><b>increases</b></span> the predicted value."
        else:
            return f" has a negative impact, and <span style='color:red'><b>decreases</b></span> the predicted value."

    df["SHAP Value Impact"] = df.index.map(lambda x: f"<b>{x}</b>") + df["SHAP Value"].apply(apply_text_explanation)

    return df

def display_lime_explanation(explanation):
    exp_list = explanation.as_list()
    exp_df = pd.DataFrame(exp_list, columns=['Feature', 'Contribution'])
    return exp_df

def parse_condition(condition):
    if '<=' in condition:
        feature, threshold = condition.split(' <= ')
        explanation = f"if the value of <b>{feature}</b> is less than or equal to <b>{threshold}</b>"
    elif '>=' in condition:
        feature, threshold = condition.split(' >= ')
        explanation = f"if the value of <b>{feature}</b> is greater than or equal to <b>{threshold}</b>"
    elif '<' in condition and '>' in condition:
        match = re.match(r'([\d.]+) < ([^ ]+) <= ([\d.]+)', condition)
        if match:
            lower, feature, upper = match.groups()
            explanation = f"if the value of <b>{feature}</b> is between <b>{lower}</b> and <b>{upper}</b>"
    elif '<=' in condition and '>' in condition:
        match = re.match(r'([\d.]+) <= ([^ ]+) < ([\d.]+)', condition)
        if match:
            lower, feature, upper = match.groups()
            explanation = f"if the value of <b>{feature}</b> is between <b>{lower}</b> and <b>{upper}</b>"
    else:
        explanation = f"if the value of <b>{condition}</b>"

    return explanation

def generate_lime_explanation_text(exp_df):
    explanation_text = []
    for index, row in exp_df.iterrows():
        condition = row['Feature']
        contribution = row['Contribution']
        impact = 'increases' if contribution > 0 else 'decreases'
        color = 'green' if contribution > 0 else 'red'
        condition_explanation = parse_condition(condition)
        explanation_text.append(
            f"{condition_explanation} <span style='color:{color}'><b>{impact}</b></span> the predicted value by {abs(contribution):.4f}"
        )
    return explanation_text

def main():
    with st.form('prediction_form'):
        st.subheader("Enter the input for following features:")

        age = st.slider("Age in years: ", 0, 100, value=50, format="%d")
        sex = st.selectbox("Sex: ", options=[1, 0], format_func=lambda x: 'Male' if x == 1 else 'Female')
        chest_pain_type = st.selectbox("Chest pain type: ", options=[1, 2, 3, 0], format_func=lambda x: {
            1: 'Typical angina',
            2: 'Atypical angina',
            3: 'Non-anginal pain',
            0: 'Asymptomatic'
        }[x])
        resting_bp = st.slider("Resting blood pressure (in mm Hg): ", 90, 200, value=120, format="%d")
        cholesterol = st.slider("Serum cholesterol in mg/dl: ", 100, 600, value=200, format="%d")
        fasting_blood_sugar = st.selectbox("Fasting blood sugar > 120 mg/dl: ", options=[1, 0], format_func=lambda x: 'True' if x == 1 else 'False')
        resting_ecg = st.selectbox("Resting electrocardiographic results: ", options=[1, 2, 0], format_func=lambda x: {
            1: 'Normal',
            2: 'ST-T wave abnormality',
            0: 'Hypertrophy'
        }[x])
        max_heart_rate = st.slider("Maximum heart rate achieved: ", 60, 220, value=140, format="%d")
        exercise_angina = st.selectbox("Exercise induced angina: ", options=[1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
        oldpeak = st.slider("ST depression induced by exercise relative to rest: ", 0.0, 10.0, value=1.0, format="%.1f")
        ST_slope = st.selectbox("Slope of the peak exercise ST segment: ", options=[0, 1, 2, 3], format_func=lambda x: {
            0: 'Normal',
            1: 'Upsloping',
            2: 'Flat',
            3: 'Down sloping'
        }[x])
        
        submit = st.form_submit_button("Predict")

    if submit:
        data = pd.DataFrame(np.array([age, sex, chest_pain_type, resting_bp, cholesterol, fasting_blood_sugar, resting_ecg,
                                      max_heart_rate, exercise_angina, oldpeak, ST_slope]).reshape(1, -1), columns=X_train.columns)
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)
        
        prediction_text = f'<span style="color:{"red" if prediction[0] == 1 else "green"}; font-size: 24px;">{"Yes" if prediction[0] == 1 else "No"}</span>'
        st.header('Prediction for User Input')
        st.markdown(f'The predicted presence of heart disease is: {prediction_text}', unsafe_allow_html=True)

        explainer = shap.Explainer(model, X_train, feature_names=X_train.columns)
        shap_values_input = explainer(data_scaled)

        def display_force_plot(shap_values, original_data):
            shap_values.data = original_data.values
            return shap_values

        shap_values_input_original = display_force_plot(shap_values_input, data)
        shap_values = explainer(X_train)
        st.header('User Input Data')
        st.write(data)
        st.header('Explanation for User Input')

        if shap_values_input.shape[1] > 0:
            left_column, right_column = st.columns([2, 1])

            with left_column:
                st.subheader("Waterfall Plot")
                st_shap(shap.plots.waterfall(shap_values_input_original[0]), height=300)

            with right_column:
                st.subheader("Waterfall Plot Explanation")
                explanation_df = get_single_explanation(shap_values_input[0].values, data)
                markdown_string = "\n".join([f"- {item}" for item in explanation_df["SHAP Value Impact"].tolist()])
                st.markdown(markdown_string, unsafe_allow_html=True)

            with st.expander("Expand for larger view of the Waterfall Plot"):
                st_shap(shap.plots.waterfall(shap_values_input_original[0]), height=600)

            st.subheader("Force Plot Explanation")
            st.markdown("""
                - **Base Value**: This is the average prediction of the model across all instances. In other words, it's what you would predict if you didn't know any features for the current output. It's the starting point of the plot.
                - **SHAP Values**: Each feature used in the model is assigned a SHAP value. This value represents how much knowing that feature changes the output of the model for the instance in question. The SHAP value for a feature is proportional to the difference between the prediction for the instance and the base value, and it's allocated according to the contribution of each feature.
                - **Color**: The features are color-coded based on their values for the specific instance. High values are shown in red, and low values are shown in blue. This provides a visual way to see which features are high or low for the given instance, and how that contributes to the prediction.
                - **Position on the X-axis**: The position of a SHAP value on the X-axis shows whether the effect of that value is associated with a higher or lower prediction. If a feature's SHAP value is positioned to the right of the base value, it means that this feature increases the prediction; if it's to the left, it decreases the prediction.
                - **Size of the SHAP Value**: The magnitude of a SHAP value tells you the importance of that feature in contributing to the difference between the actual prediction and the base value. Larger SHAP values (either positive or negative) have a bigger impact.
            """)

            st.subheader("Force Plot")
            st_shap(shap.plots.force(shap_values_input_original[0]), height=300)

        else:
            st.write("Error: SHAP values input has no features to plot.")

        st.header('Summary and Beeswarm Plots for All Features')
        st.subheader("Summary Plot")
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 5)
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        st.pyplot(fig)

        st.subheader("Beeswarm Plot")
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 5)
        shap.summary_plot(shap_values, X_train, show=False)
        st.pyplot(fig)

        st.header('LIME Explanation')
        lime_explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['No Heart Disease', 'Heart Disease'], discretize_continuous=True)
        lime_exp = lime_explainer.explain_instance(data_scaled[0], model.predict_proba, num_features=11)

        lime_df = display_lime_explanation(lime_exp)

        st.subheader("LIME Explanation")
        st.dataframe(lime_df)

        left_column, right_column = st.columns([2, 1])

        with left_column:
            st.subheader("LIME Explanation Plot")
            fig = lime_exp.as_pyplot_figure()
            fig.set_size_inches(10, 5)  # Adjust the size as needed
            st.pyplot(fig)

        with right_column:
            st.subheader("LIME Explanation Details")
            lime_explanation_text = generate_lime_explanation_text(lime_df)
            markdown_string = "\n".join([f"- {item}" for item in lime_explanation_text])
            st.markdown(markdown_string, unsafe_allow_html=True)

        html_path = "lime_explanation.html"
        with open(html_path, "w") as file:
            file.write(lime_exp.as_html())
        with open(html_path, 'r') as file:
            html_content = file.read()
        
        st.components.v1.html(html_content, height=800, scrolling=True)

# FastAPI setup
app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    print(f"Received data: {data}")
    try:
        age = data['age']
        sex = data['sex']
        chest_pain_type = data['chest_pain_type']
        resting_bp = data['resting_bp']
        cholesterol = data['cholesterol']
        fasting_blood_sugar = data['fasting_blood_sugar']
        resting_ecg = data['resting_ecg']
        max_heart_rate = data['max_heart_rate']
        exercise_angina = data['exercise_angina']
        oldpeak = data['oldpeak']
        ST_slope = data['ST_slope']
    except KeyError as e:
        return {"error": f"Missing key: {e}"}

    data_df = pd.DataFrame(np.array([age, sex, chest_pain_type, resting_bp, cholesterol, fasting_blood_sugar,
                                     resting_ecg, max_heart_rate, exercise_angina, oldpeak, ST_slope]).reshape(1, -1),
                           columns=X_train.columns)
    data_scaled = scaler.transform(data_df)
    prediction = model.predict(data_scaled)
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    import threading
    threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")).start()
    main()
