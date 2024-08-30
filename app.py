import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from fuzzywuzzy import fuzz

# Load your pre-trained models
def load_ml_models():
    rf_model = joblib.load("models/model.pkl")
    xgb_model = joblib.load("models/cv.pkl")
    return rf_model, xgb_model

def load_dl_models():
    lstm_model = tf.keras.models.load_model("models/lstm_model.h5")
    bilstm_model = tf.keras.models.load_model("models/bilstm_model.h5")
    gru_model = tf.keras.models.load_model("models/gru_model.h5")
    return lstm_model, bilstm_model, gru_model

# Load the CountVectorizer fitted on the training data
def load_vectorizer():
    vectorizer = joblib.load("models/vectorizer.pkl")
    return vectorizer

# Define missing functions
def common_words(row):
    q1_words = set(row['question1'].split())
    q2_words = set(row['question2'].split())
    return len(q1_words & q2_words)

def total_words(row):
    q1_words = set(row['question1'].split())
    q2_words = set(row['question2'].split())
    return len(q1_words) + len(q2_words)

def fetch_token_features(row):
    q1 = row['question1'].split()
    q2 = row['question2'].split()
    
    if len(q1) == 0 or len(q2) == 0:
        return [0] * 8
    
    cwc_min = len(set(q1).intersection(set(q2))) / min(len(set(q1)), len(set(q2)))
    cwc_max = len(set(q1).intersection(set(q2))) / max(len(set(q1)), len(set(q2)))
    
    csc_min = len(set(' '.join(q1)).intersection(set(' '.join(q2)))) / min(len(set(' '.join(q1))), len(set(' '.join(q2))))
    csc_max = len(set(' '.join(q1)).intersection(set(' '.join(q2)))) / max(len(set(' '.join(q1))), len(set(' '.join(q2))))
    
    ctc_min = sum(1 for c in q1 if c in q2) / min(len(q1), len(q2))
    ctc_max = sum(1 for c in q1 if c in q2) / max(len(q1), len(q2))
    
    last_word_eq = int(q1[-1] == q2[-1])
    first_word_eq = int(q1[0] == q2[0])
    
    return [cwc_min, cwc_max, csc_min, csc_max, ctc_min, ctc_max, last_word_eq, first_word_eq]

def fetch_length_features(row):
    len_q1 = len(row['question1'])
    len_q2 = len(row['question2'])
    
    abs_len_diff = abs(len_q1 - len_q2)
    mean_len = (len_q1 + len_q2) / 2
    lcs = len(set(row['question1']).intersection(set(row['question2'])))  # Longest Common Substring ratio
    
    return [abs_len_diff, mean_len, lcs]

def fetch_fuzzy_features(row):
    fuzz_ratio = fuzz.ratio(row['question1'], row['question2'])
    fuzz_partial_ratio = fuzz.partial_ratio(row['question1'], row['question2'])
    token_sort_ratio = fuzz.token_sort_ratio(row['question1'], row['question2'])
    token_set_ratio = fuzz.token_set_ratio(row['question1'], row['question2'])
    
    return [fuzz_ratio, fuzz_partial_ratio, token_sort_ratio, token_set_ratio]

# Preprocessing function (you can modify this as needed)
def preprocess(q):
    # Example preprocessing steps
    q = q.lower()
    # Other preprocessing steps
    return q

# Streamlit interface
st.title("Question Similarity Predictor")

question1 = st.text_input("Enter Question 1")
question2 = st.text_input("Enter Question 2")

model_type = st.radio("Choose Model Type", ("Machine Learning", "Deep Learning"))

if model_type == "Machine Learning":
    ml_model_choice = st.selectbox("Choose ML Model", ("Random Forest", "XGBoost"))
elif model_type == "Deep Learning":
    dl_model_choice = st.selectbox("Choose DL Model", ("LSTM", "BiLSTM", "GRU"))

if st.button("Evaluate"):
    st.write("Button clicked, starting evaluation...")

    # Preprocess questions
    question1_processed = preprocess(question1)
    question2_processed = preprocess(question2)
    
    st.write(f"Preprocessed Question 1: {question1_processed}")
    st.write(f"Preprocessed Question 2: {question2_processed}")
    
    # Combine into a DataFrame
    input_df = pd.DataFrame([[question1_processed, question2_processed]], columns=["question1", "question2"])
    st.write("Input DataFrame:", input_df)

    # Apply feature extraction
    input_df["q1_len"] = input_df["question1"].str.len()
    input_df["q2_len"] = input_df["question2"].str.len()

    input_df["q1_num_words"] = input_df["question1"].apply(lambda row: len(row.split(" ")))
    input_df["q2_num_words"] = input_df["question2"].apply(lambda row: len(row.split(" ")))

    input_df["common_words"] = input_df.apply(common_words, axis=1)
    input_df["total_words"] = input_df.apply(total_words, axis=1)
    input_df["word_share"] = round(input_df["common_words"] / input_df['total_words'] * 100, 2)
    
    st.write("Features after basic processing:", input_df)
    
    # Token features
    token_features = input_df.apply(fetch_token_features, axis=1)
    input_df["cwc_min"] = list(map(lambda x : x[0], token_features))
    input_df["cwc_max"] = list(map(lambda x : x[1], token_features))
    input_df["csc_min"] = list(map(lambda x : x[2], token_features))
    input_df["csc_max"] = list(map(lambda x : x[3], token_features))
    input_df["ctc_min"] = list(map(lambda x : x[4], token_features))
    input_df["ctc_max"] = list(map(lambda x : x[5], token_features))
    input_df["last_word_eq"] = list(map(lambda x : x[6], token_features))

    st.write("Features after token processing:", input_df)
    
    # Load the vectorizer and transform the input questions
    vectorizer = load_vectorizer()
    q1_arr = vectorizer.transform([question1_processed]).toarray()
    q2_arr = vectorizer.transform([question2_processed]).toarray()
    
    temp_df1 = pd.DataFrame(q1_arr)
    temp_df2 = pd.DataFrame(q2_arr)
    temp_df = pd.concat([temp_df1, temp_df2], axis=1)
    
    st.write("Vectorized features:", temp_df)

    # Combine all features
    X_features = np.hstack([input_df.values, temp_df.values])
    st.write("Combined features:", X_features)

    if model_type == "Machine Learning":
        rf_model, xgb_model = load_ml_models()
        st.write("Loaded Machine Learning models")
        if ml_model_choice == "Random Forest":
            prediction = rf_model.predict(X_features)
        elif ml_model_choice == "XGBoost":
            prediction = xgb_model.predict(X_features)
    
    elif model_type == "Deep Learning":
        lstm_model, bilstm_model, gru_model = load_dl_models()
        st.write("Loaded Deep Learning models")
        if dl_model_choice == "LSTM":
            prediction = lstm_model.predict(X_features)
        elif dl_model_choice == "BiLSTM":
            prediction = bilstm_model.predict(X_features)
        elif dl_model_choice == "GRU":
            prediction = gru_model.predict(X_features)
    
    # Display prediction
    st.write(f"Predicted Output: {prediction}")
