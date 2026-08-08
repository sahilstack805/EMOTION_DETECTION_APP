
import streamlit as st
import joblib
import string
import re
import nltk

# Download required NLTK data
try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))


# =========================================================
# LOAD SAVED MODEL FILES
# =========================================================

tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
logistic_model = joblib.load("logistic_model.pkl")
emotion_numbers = joblib.load("emotion_mapping.pkl")


# =========================================================
# TEXT PREPROCESSING
# =========================================================

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = text.split()
    words = [
        word for word in words
        if word not in stop_words
    ]

    text = " ".join(words)

    return text


# =========================================================
# EMOTION MAPPING
# =========================================================

# emotion_numbers was saved as:
# emotion -> number
#
# Example:
# {
#     "sadness": 0,
#     "joy": 1,
#     ...
# }

# Create reverse mapping:
# number -> emotion
reverse_emotion_mapping = {
    value: key
    for key, value in emotion_numbers.items()
}


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_emotion(text):

    # Preprocess input
    cleaned_text = preprocess_text(text)

    # Convert text into TF-IDF features
    text_tfidf = tfidf_vectorizer.transform([cleaned_text])

    # Predict emotion number
    prediction = logistic_model.predict(text_tfidf)[0]

    # Convert number to emotion name
    emotion = reverse_emotion_mapping.get(
        prediction,
        str(prediction)
    )

    return emotion


# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="centered"
)


# =========================================================
# TITLE
# =========================================================

st.title("😊 Emotion Detection")
st.write(
    "Enter a sentence and the machine learning model "
    "will predict the emotion expressed in the text."
)


# =========================================================
# TEXT INPUT
# =========================================================

user_text = st.text_area(
    "Enter your text:",
    placeholder="Example: I am very happy today!",
    height=150
)


# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button("Predict Emotion"):

    if user_text.strip() == "":
        st.warning("Please enter some text.")

    else:

        emotion = predict_emotion(user_text)

        st.success(f"Predicted Emotion: **{emotion.upper()}**")


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Emotion Detection using TF-IDF and Logistic Regression"
)

