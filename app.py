from flask import Flask, render_template_string, request, jsonify
import random
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
from difflib import get_close_matches

# ------------------ Load Data ------------------
training = pd.read_csv('Data/Training.csv')
training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
training = training.loc[:, ~training.columns.duplicated()]

cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

# Encode target
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(x_train, y_train)

symptoms_dict = {symptom: idx for idx, symptom in enumerate(x)}

# ------------------ Symptom Extractor ------------------
symptom_synonyms = {
    "stomach ache": "stomach_pain",
    "belly pain": "stomach_pain",
    "tummy pain": "stomach_pain",
    "loose motion": "diarrhea",
    "motions": "diarrhea",
    "high temperature": "fever",
    "temperature": "fever",
    "feaver": "fever",
    "coughing": "cough",
    "throat pain": "sore_throat",
    "cold": "chills",
    "breathing issue": "breathlessness",
    "shortness of breath": "breathlessness",
    "body ache": "muscle_pain",
}

def extract_symptoms(user_input, all_symptoms):
    extracted = []
    text = user_input.lower().replace("-", " ")

    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            extracted.append(mapped)

    for symptom in all_symptoms:
        if symptom.replace("_", " ") in text:
            extracted.append(symptom)

    words = re.findall(r"\w+", text)
    for word in words:
        close = get_close_matches(word, [s.replace("_", " ") for s in all_symptoms], n=1, cutoff=0.8)
        if close:
            for sym in all_symptoms:
                if sym.replace("_", " ") == close[0]:
                    extracted.append(sym)

    return list(set(extracted))

def predict_disease(symptoms_list):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    pred_proba = model.predict_proba([input_vector])[0]
    pred_class = np.argmax(pred_proba)
    disease = le.inverse_transform([pred_class])[0]
    confidence = round(pred_proba[pred_class] * 100, 2)
    return disease, confidence

# ------------------ Flask App ------------------
app = Flask(__name__)

# Simple chat HTML
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Health Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f7f9fc; }
        #chatbox { width: 500px; margin: auto; border: 1px solid #ccc; background: white; border-radius: 10px; padding: 20px; }
        .bot, .user { margin: 10px 0; padding: 10px; border-radius: 8px; max-width: 80%; }
        .bot { background: #e3f2fd; text-align: left; }
        .user { background: #c8e6c9; text-align: right; margin-left: auto; }
        #messages { height: 300px; overflow-y: auto; border-bottom: 1px solid #ccc; padding-bottom: 10px; }
        input { width: 80%; padding: 10px; }
        button { padding: 10px; }
    </style>
</head>
<body>
    <div id="chatbox">
        <h3>🤖 Health Chatbot</h3>
        <div id="messages"></div>
        <input type="text" id="userInput" placeholder="Type your message..." autofocus>
        <button onclick="sendMessage()">Send</button>
    </div>

<script>
function addMessage(sender, text) {
    let messages = document.getElementById("messages");
    let div = document.createElement("div");
    div.className = sender;
    div.textContent = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

function sendMessage() {
    let input = document.getElementById("userInput");
    let text = input.value.trim();
    if (text === "") return;

    addMessage("user", text);
    input.value = "";

    fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text })
    })
    .then(res => res.json())
    .then(data => {
        addMessage("bot", data.reply);
    });
}
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

# Simple chatbot state
user_symptoms = []

@app.route("/chat", methods=["POST"])
def chat():
    global user_symptoms
    data = request.get_json()
    user_input = data.get("message", "")

    if not user_symptoms:  # first message
        symptoms_list = extract_symptoms(user_input, cols)
        if symptoms_list:
            user_symptoms = symptoms_list
            return jsonify({"reply": f"✅ I detected symptoms: {', '.join(symptoms_list)}. How many days have you had these?"})
        else:
            return jsonify({"reply": "❌ Sorry, I couldn't detect any valid symptoms. Please try again."})
    elif len(user_symptoms) > 0 and "days" not in user_symptoms:
        user_symptoms.append("days")  # just marking that days were answered
        return jsonify({"reply": "👍 Thanks! On a scale of 1–10, how severe are your symptoms?"})
    elif "days" in user_symptoms and "severity" not in user_symptoms:
        user_symptoms.append("severity")
        disease, confidence = predict_disease([s for s in user_symptoms if s not in ["days","severity"]])
        reply = f"🩺 You may have **{disease}** (confidence {confidence}%).\n💡 Please consult a doctor for confirmation."
        user_symptoms = []  # reset
        return jsonify({"reply": reply})

    return jsonify({"reply": "🤔 Tell me more about your symptoms."})

if __name__ == "__main__":
    app.run(debug=True)
