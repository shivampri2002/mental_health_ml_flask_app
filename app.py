from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from flask_cors import CORS

from tensorflow.keras.models import load_model

from utils import ltsm_model_input_transformer


import joblib


# Load the model
modelLTSM = load_model('../my_model_ltsm.h5')
modelXG = joblib.load('../XGBoost_model.pkl')

app = Flask(__name__)
CORS(app)


# Load the preprocessor and label encoder used during training
preprocessor_xg = joblib.load('../preprocessor_xg.pkl')
label_encoder_xg = joblib.load('../label_encoder_xg.pkl')
# Load the vectorizer
loaded_vectorizer_ltsm = joblib.load('../text_vectorizer_ltsm.pkl')


#xgboost categorical columns
categorical_cols_xg = ['Gender', 'Country', 'Occupation', 'self_employed', 'family_history', 'treatment', 'Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Mental_Health_History', 'Coping_Struggles', 'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options']

#ltsm labels
original_ltsm_labels = ['Suicidal', 'Depression', 'Personality disorder', 'Stress', 'Normal', 'Anxiety', 'Bipolar']

index_to_label_ltsm = {index: label for index, label in enumerate(original_ltsm_labels)}


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the POST request
    data = request.json  # Assuming data is sent as JSON
    # input_data = np.array(data['input'])  
    # Assuming input is in dictionary form with cols as key and its values
    # Convert to DataFrame
    survey_df = pd.DataFrame([data['input']])
    survey_df_no_target = survey_df[categorical_cols_xg]
    survey_encoded = preprocessor_xg.transform(survey_df_no_target)

    # Make predictions using the XGBoost model
    predictionsXG = modelXG.predict(survey_encoded)  

    #output is Low, Medium, High
    predictionsXG = label_encoder_xg.inverse_transform(predictionsXG)

    #ltsm model
    explain_input = pd.DataFrame([data["explain_input"]])
    explain_input = ltsm_model_input_transformer(explain_input)
    vectorized_explain_input = loaded_vectorizer_ltsm(explain_input.values)

    predictionsLTSM = modelLTSM.predict(vectorized_explain_input)
    print(predictionsLTSM)

    predictionsLTSM = np.argmax(predictionsLTSM, axis=1)
    print(predictionsLTSM)

    decoded_predictionsLTSM = index_to_label_ltsm[predictionsLTSM[0]]


    # Return the predictions as a JSON response
    return jsonify({"xg_res": ''.join(predictionsXG.tolist()), "ltsm_res": decoded_predictionsLTSM,})

if __name__ == '__main__':
    app.run(debug=True)
