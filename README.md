# Medicine Recommendation System

An AI-powered Flask web application that predicts possible diseases from user-entered symptoms and provides supporting health guidance, including precautions, medicines, diets, and workout suggestions.

## Features

- Symptom-based disease prediction using a trained Support Vector Classifier model.
- Medicine, precaution, diet, and workout recommendations for predicted diseases.
- Clean Flask web interface with responsive Bootstrap styling.
- Symptom suggestions and quick-select symptom chips.
- Speech recognition support in compatible browsers.
- Contact, about, developer, and blog pages.

## Tech Stack

- Python
- Flask
- NumPy
- Pandas
- scikit-learn
- Bootstrap

## Project Structure

```text
Medicine-Recommendation-System/
|-- main.py
|-- constants.py
|-- requirements.txt
|-- svc.pkl
|-- Training.csv
|-- description.csv
|-- diets.csv
|-- medications.csv
|-- precautions_df.csv
|-- symtoms_df.csv
|-- workout_df.csv
|-- static/
|   |-- style.css
|   |-- img.png
|   `-- neeraj-profile-anime.png
`-- templates/
    |-- index.html
    |-- about.html
    |-- contact.html
    |-- developer.html
    `-- blog.html
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/neerajsinghyadav3434/medicine-recommendation-system.git
cd medicine-recommendation-system
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python main.py
```

Open the app in your browser:

```text
http://127.0.0.1:5000
```

## Usage

1. Enter symptoms separated by commas.
2. Use symptom suggestions or quick symptom chips for common inputs.
3. Submit the form to get a predicted disease.
4. View recommendations for description, precautions, medications, workouts, and diet.

Example symptoms:

```text
itching, skin rash
fever, cough
headache, nausea
```

## Security Notes

- No API keys, tokens, passwords, or secret files are required by this project.
- Local development folders such as `.venv`, `.uv-cache`, `.uv-python`, and `__pycache__` are ignored by Git.
- Flask debug mode is disabled by default. To enable it locally, run with `FLASK_DEBUG=1`.
- `svc.pkl` is a Python pickle model file. Only use this model file from trusted sources because pickle files can execute code during loading.
- The app provides informational guidance only and is not a replacement for professional medical advice.

## Disclaimer

This project is for educational purposes only. Disease predictions and recommendations should not be treated as a medical diagnosis. Always consult a qualified healthcare professional before taking medication or starting any treatment.

## Author

Neeraj Singh Yadav

- LinkedIn: https://www.linkedin.com/in/neeraj-singh-yadav/
- Email: neerajsinghyadav9271@gmail.com
