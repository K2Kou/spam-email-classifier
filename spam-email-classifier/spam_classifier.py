# -----------------------------
# Spam Email Classifier (MCA Project)
# -----------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import streamlit as st

# -----------------------------
# Load and Train Model (only once)
# -----------------------------
@st.cache_resource
def load_model():
    data = pd.read_csv("spam.csv")

    # Keep required columns
    data = data[['label', 'message']]
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    X = data['message']
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_vec))

    return model, vectorizer, accuracy

model, vectorizer, accuracy = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📧 Spam Email Classifier")

st.write(f"Model Accuracy: {accuracy:.2f}")

# Unique key added ✅
input_msg = st.text_input("Enter your message:", key="input_message")

if st.button("Check"):
    if input_msg.strip() == "":
        st.warning("Please enter a message")
    else:
        vec = vectorizer.transform([input_msg])
        result = model.predict(vec)

        if result[0] == 1:
            st.error("⚠️ This is Spam")
        else:
            st.success("✅ This is Not Spam")