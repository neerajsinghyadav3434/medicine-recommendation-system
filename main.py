import os
from pathlib import Path
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
from constants import symptoms_dict, diseases_list, synonyms_map

# Project paths
BASE_DIR = Path(__file__).resolve().parent

# Initialize Flask app
app = Flask(__name__)

# Load datasets
sym_des = pd.read_csv(BASE_DIR / "symtoms_df.csv")
precautions = pd.read_csv(BASE_DIR / "precautions_df.csv")
workout = pd.read_csv(BASE_DIR / "workout_df.csv")
description = pd.read_csv(BASE_DIR / "description.csv")
medications = pd.read_csv(BASE_DIR / "medications.csv")
diets = pd.read_csv(BASE_DIR / "diets.csv")

# Load model
with open(BASE_DIR / "svc.pkl", "rb") as model_file:
    svc = pickle.load(model_file)

symptom_options = sorted(symptom.replace("_", " ") for symptom in symptoms_dict)


def helper(dis):
    """
    Helper function to fetch details for a predicted disease.
    Returns description, precautions, medications, diets, and workouts.
    """
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc]) or "No description available."

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = pre.values.tolist()

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values] or ["No medication information available."]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values] or ["No diet information available."]

    wrkout = workout[workout['disease'] == dis]['workout']
    wrkout = [item for item in wrkout.values] or ["No workout information available."]

    return desc, pre, med, die, wrkout


def normalize_symptoms(symptoms_text):
    """
    Converts comma-separated user input into known symptom keys.
    Returns recognized symptoms and invalid symptoms separately.
    """
    raw_symptoms = [s.strip() for s in symptoms_text.split(",")]
    normalized_symptoms = []

    for symptom in raw_symptoms:
        symptom = symptom.strip("[]'\" ").lower().replace("-", " ").replace(" ", "_")
        if not symptom:
            continue
        normalized_symptoms.append(synonyms_map.get(symptom, symptom))

    unique_symptoms = list(dict.fromkeys(normalized_symptoms))
    invalid_symptoms = [symptom for symptom in unique_symptoms if symptom not in symptoms_dict]

    return unique_symptoms, invalid_symptoms


def get_predicted_value(patient_symptoms):
    """
    Predicts the disease based on the input symptoms vector.
    """
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    input_frame = pd.DataFrame([input_vector], columns=symptoms_dict.keys())
    return diseases_list[svc.predict(input_frame)[0]]


# Routes
@app.route("/")
def index():
    return render_template("index.html", symptom_options=symptom_options)

@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        
        if not symptoms or symptoms.strip().lower() == "symptoms":
            message = "Please enter at least one symptom, separated by commas."
            return render_template('index.html', message=message, symptom_options=symptom_options)

        user_symptoms, invalid_symptoms = normalize_symptoms(symptoms)
        
        if not user_symptoms:
            message = "Please enter at least one symptom, separated by commas."
            return render_template('index.html', message=message, symptom_options=symptom_options)

        if invalid_symptoms:
            readable_symptoms = ", ".join(symptom.replace("_", " ") for symptom in invalid_symptoms)
            message = f"Sorry, I don't recognize these symptoms: {readable_symptoms}. Please check spelling or choose from the suggestions."
            return render_template('index.html', message=message, symptom_options=symptom_options)

        try:
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            my_precautions = precautions[0] if precautions else ["No precaution information available."]

            return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                                   workout=workout, symptom_options=symptom_options)
        except Exception as e:
            message = "An unexpected error occurred. Please try again."
            app.logger.exception("Prediction failed")
            return render_template('index.html', message=message, symptom_options=symptom_options)

    return render_template('index.html', symptom_options=symptom_options)


@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/developer')
def developer():
    return render_template("developer.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")


if __name__ == '__main__':
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode)
