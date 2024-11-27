
from bs4 import BeautifulSoup
import requests
import re
import streamlit as st
import io
import sys
import distance
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
from bs4 import BeautifulSoup
import requests
import re

def predict_RF(q1,q2):

    my_dict = {}

    def preprocess(q):

        q=str(q).lower().strip()
        q=BeautifulSoup(q)
        q=q.get_text()

        q=q.replace("%"," percent")
        q=q.replace("$"," dollar ")
        q=q.replace("₹"," ruppees ")
        q=q.replace("€"," euro ")
        q=q.replace("@"," at ")
        q=q.replace("&"," and ")

        q=q.replace("[math]","")

        q=q.replace(",000,000,000","b")
        q=q.replace(",000,000","m")
        q=q.replace(",000","k")
        q=re.sub(r"([0-9]+)000000000",r"\1b",q)
        q=re.sub(r"([0-9]+)000000",r"\1m",q)
        q=re.sub(r"([0-9]+)000",r"\1k",q)
        contractions= {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it had",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there had",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'alls": "you alls",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had",
        "you'd've": "you would have",
        "you'll": "you you will",
        "you'll've": "you you will have",
        "you're": "you are",
        "you've": "you have"
    }
        q_decontracted=[]

        for word in q.split():
            if word in contractions:

                word=contractions[word]

        q_decontracted.append(word)


        q=" ".join(q_decontracted)
        q=str(q).lower().strip()
        q=q.replace("'ve"," have")
        q=q.replace("n't"," not")
        q=q.replace("'re"," are")
        q=q.replace("'ll"," will")


        pattern=re.compile("\W")
        q=re.sub(pattern," ",q).strip()

        return q

        
        
    import nltk
    nltk.download('stopwords')

    from nltk.corpus import stopwords


    def fetch_token_features_for_two_questions(q1, q2):
        STOP_WORDS = stopwords.words("english")
        SAFE_DIV = 0.0001

        # Initialize token features as a dictionary
        token_features = {}

        # Tokenize the questions
        q1_tokens = q1.split(" ")
        q2_tokens = q2.split(" ")

        # Check for empty tokens
        if len(q1_tokens) == 0 or len(q2_tokens) == 0:
            return token_features

        # Unique non-stop words
        q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
        q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

        # Unique stop words
        q1_stop = set([word for word in q1_tokens if word in STOP_WORDS])
        q2_stop = set([word for word in q2_tokens if word in STOP_WORDS])

        # Calculate common word counts
        common_word_count = len(q1_words.intersection(q2_words))
        common_stop_count = len(q1_stop.intersection(q2_stop))
        common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

        # Calculate features
        token_features["cwc_min"] = common_word_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features["cwc_max"] = common_word_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features["csc_min"] = common_stop_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features["csc_max"] = common_stop_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features["ctc_min"] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features["ctc_max"] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features["last_word_eq"] = int(q1_tokens[-1] == q2_tokens[-1])  # Check if last words are the same
        token_features["first_word_eq"] = int(q1_tokens[0] == q2_tokens[0])    # Check if first words are the same

        return token_features

    import distance

    def fetch_length_features_for_two_questions(q1, q2):
        # Tokenize the input strings
        q1_tokens = q1.split(" ")
        q2_tokens = q2.split(" ")

        # Initialize a dictionary to store the length features
        length_features = {}

        # Return length features if any string is empty
        if len(q1_tokens) == 0 or len(q2_tokens) == 0:
            return length_features

        # Absolute length difference
        length_features["abs_len_diff"] = abs(len(q1_tokens) - len(q2_tokens))

        # Average length
        length_features["mean_len"] = (len(q1_tokens) + len(q2_tokens)) / 2

        # Longest common substring ratio
        strs = list(distance.lcsubstrings(q1, q2))
        try:
            length_features["longest_substr_ratio"] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
        except:
            length_features["longest_substr_ratio"] = 0

        return length_features


    from fuzzywuzzy import fuzz



    def fetch_fuzzy_features_for_two_questions(q1, q2):
        # Initialize a dictionary to store fuzzy features
        fuzzy_features = {}

        # Calculate fuzzy matching features and store them in the dictionary
        fuzzy_features["fuzz_ratio"] = fuzz.QRatio(q1, q2)  # Full ratio
        fuzzy_features["fuzz_partial_ratio"] = fuzz.partial_ratio(q1, q2)  # Partial ratio
        fuzzy_features["token_sort_ratio"] = fuzz.token_sort_ratio(q1, q2)  # Token sort ratio
        fuzzy_features["token_set_ratio"] = fuzz.token_set_ratio(q1, q2)  # Token set ratio

        return fuzzy_features

    def totalword(question1, question2):
        w1 = set(map(lambda word: word.lower().strip(), question1.split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), question2.split(" ")))
        return len(w1) + len(w2)


    def doit(q1,q2):
        q1_preprocess = preprocess(q1)
        q2_preprocess = preprocess(q2)
        # print(q1_preprocess)
        # print(q2_preprocess)



        q1_len = len(q1_preprocess)
        q2_len = len(q2_preprocess)

        my_dict["q1_len"] = q1_len
        my_dict["q2_len"] = q2_len



        q1_num_of_words = len(q1_preprocess.split())
        q2_num_of_words = len(q2_preprocess.split())

        my_dict["q1_num_of_words"] = q1_num_of_words
        my_dict["q2_num_of_words"] = q2_num_of_words




        q1_words = set(q1_preprocess.lower().split())
        q2_words = set(q2_preprocess.lower().split())

        # Find the intersection of the two sets
        common_words = len(q1_words.intersection(q2_words))

        my_dict["common_words"] = common_words



        
        total_words = totalword(q1_preprocess, q2_preprocess)
        my_dict["total_words"] = total_words



        word_share = round((common_words / total_words) * 100, 2)
        my_dict["word_share"] = word_share



        features = fetch_token_features_for_two_questions(q1_preprocess, q2_preprocess)
        # print(features)
        my_dict.update(features)

    # Fetch length features for the two questions
        length_features = fetch_length_features_for_two_questions(q1_preprocess, q2_preprocess)
        my_dict.update(length_features)



        # Fetch fuzzy features for the two questions
        fuzzy_features = fetch_fuzzy_features_for_two_questions(q1_preprocess, q2_preprocess)
        # print(fuzzy_features)
        my_dict.update(fuzzy_features)


    doit(q1,q2)

    import joblib
    import pickle
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    # preprocessing and feature calculation function

    # Load the model and scaler with joblib if it was saved
    vectorizer = joblib.load('count_vectorizer1.pkl')
    model = joblib.load('random_forest_model.pkl')


    # Use the CountVectorizer to transform the text of the new questions
    q1_vectorized = vectorizer.transform([q1]).toarray()
    q2_vectorized = vectorizer.transform([q2]).toarray()


    # Combine the vectorized features (just once)
    combined_vectorized_features = np.concatenate((q1_vectorized, q2_vectorized), axis=1)

    # Convert dictionary to numpy array and reshape
    manually_features = np.array(list(my_dict.values())).reshape(1, -1)

    # Scale the manually calculated features using the loaded scaler
    # scaled_features = scaler.transform(manually_features)

    # Combine both sets of features in the correct order
    final_features = np.concatenate((combined_vectorized_features,manually_features), axis=1)

    # Pass the final features to the model for prediction
    probabilities = model.predict_proba(final_features)

    # # Example probabilities list [x, y]
    # probabilities = [0.7, 0.3]  # You can change this to test with different values

    # Extract values of x and y
    x = probabilities[0][0]
    y = probabilities[0][1]
    # Apply the condition
    if x > 0.5:
        predicted_class = 0  # If x > 0.5, return 0
    elif y > 0.5:
        predicted_class = 1  # If y > 0.5, return 1
   
    
    return predicted_class

def predict_XG(q1,q2):

    my_dict = {}

    def preprocess(q):

        q=str(q).lower().strip()
        q=BeautifulSoup(q)
        q=q.get_text()

        q=q.replace("%"," percent")
        q=q.replace("$"," dollar ")
        q=q.replace("₹"," ruppees ")
        q=q.replace("€"," euro ")
        q=q.replace("@"," at ")
        q=q.replace("&"," and ")

        q=q.replace("[math]","")

        q=q.replace(",000,000,000","b")
        q=q.replace(",000,000","m")
        q=q.replace(",000","k")
        q=re.sub(r"([0-9]+)000000000",r"\1b",q)
        q=re.sub(r"([0-9]+)000000",r"\1m",q)
        q=re.sub(r"([0-9]+)000",r"\1k",q)
        contractions= {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it had",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there had",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'alls": "you alls",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had",
        "you'd've": "you would have",
        "you'll": "you you will",
        "you'll've": "you you will have",
        "you're": "you are",
        "you've": "you have"
    }
        q_decontracted=[]

        for word in q.split():
            if word in contractions:

                word=contractions[word]

        q_decontracted.append(word)


        q=" ".join(q_decontracted)
        q=str(q).lower().strip()
        q=q.replace("'ve"," have")
        q=q.replace("n't"," not")
        q=q.replace("'re"," are")
        q=q.replace("'ll"," will")


        pattern=re.compile("\W")
        q=re.sub(pattern," ",q).strip()

        return q

        
        
    import nltk
    nltk.download('stopwords')

    from nltk.corpus import stopwords


    def fetch_token_features_for_two_questions(q1, q2):
        STOP_WORDS = stopwords.words("english")
        SAFE_DIV = 0.0001

        # Initialize token features as a dictionary
        token_features = {}

        # Tokenize the questions
        q1_tokens = q1.split(" ")
        q2_tokens = q2.split(" ")

        # Check for empty tokens
        if len(q1_tokens) == 0 or len(q2_tokens) == 0:
            return token_features

        # Unique non-stop words
        q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
        q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

        # Unique stop words
        q1_stop = set([word for word in q1_tokens if word in STOP_WORDS])
        q2_stop = set([word for word in q2_tokens if word in STOP_WORDS])

        # Calculate common word counts
        common_word_count = len(q1_words.intersection(q2_words))
        common_stop_count = len(q1_stop.intersection(q2_stop))
        common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

        # Calculate features
        token_features["cwc_min"] = common_word_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features["cwc_max"] = common_word_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features["csc_min"] = common_stop_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features["csc_max"] = common_stop_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features["ctc_min"] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features["ctc_max"] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features["last_word_eq"] = int(q1_tokens[-1] == q2_tokens[-1])  # Check if last words are the same
        token_features["first_word_eq"] = int(q1_tokens[0] == q2_tokens[0])    # Check if first words are the same

        return token_features

    import distance

    def fetch_length_features_for_two_questions(q1, q2):
        # Tokenize the input strings
        q1_tokens = q1.split(" ")
        q2_tokens = q2.split(" ")

        # Initialize a dictionary to store the length features
        length_features = {}

        # Return length features if any string is empty
        if len(q1_tokens) == 0 or len(q2_tokens) == 0:
            return length_features

        # Absolute length difference
        length_features["abs_len_diff"] = abs(len(q1_tokens) - len(q2_tokens))

        # Average length
        length_features["mean_len"] = (len(q1_tokens) + len(q2_tokens)) / 2

        # Longest common substring ratio
        strs = list(distance.lcsubstrings(q1, q2))
        try:
            length_features["longest_substr_ratio"] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
        except:
            length_features["longest_substr_ratio"] = 0

        return length_features


    from fuzzywuzzy import fuzz



    def fetch_fuzzy_features_for_two_questions(q1, q2):
        # Initialize a dictionary to store fuzzy features
        fuzzy_features = {}

        # Calculate fuzzy matching features and store them in the dictionary
        fuzzy_features["fuzz_ratio"] = fuzz.QRatio(q1, q2)  # Full ratio
        fuzzy_features["fuzz_partial_ratio"] = fuzz.partial_ratio(q1, q2)  # Partial ratio
        fuzzy_features["token_sort_ratio"] = fuzz.token_sort_ratio(q1, q2)  # Token sort ratio
        fuzzy_features["token_set_ratio"] = fuzz.token_set_ratio(q1, q2)  # Token set ratio

        return fuzzy_features

    def totalword(question1, question2):
        w1 = set(map(lambda word: word.lower().strip(), question1.split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), question2.split(" ")))
        return len(w1) + len(w2)


    def doit(q1,q2):
        q1_preprocess = preprocess(q1)
        q2_preprocess = preprocess(q2)
        # print(q1_preprocess)
        # print(q2_preprocess)



        q1_len = len(q1_preprocess)
        q2_len = len(q2_preprocess)

        my_dict["q1_len"] = q1_len
        my_dict["q2_len"] = q2_len



        q1_num_of_words = len(q1_preprocess.split())
        q2_num_of_words = len(q2_preprocess.split())

        my_dict["q1_num_of_words"] = q1_num_of_words
        my_dict["q2_num_of_words"] = q2_num_of_words




        q1_words = set(q1_preprocess.lower().split())
        q2_words = set(q2_preprocess.lower().split())

        # Find the intersection of the two sets
        common_words = len(q1_words.intersection(q2_words))

        my_dict["common_words"] = common_words



        
        total_words = totalword(q1_preprocess, q2_preprocess)
        my_dict["total_words"] = total_words



        word_share = round((common_words / total_words) * 100, 2)
        my_dict["word_share"] = word_share



        features = fetch_token_features_for_two_questions(q1_preprocess, q2_preprocess)
        # print(features)
        my_dict.update(features)

    # Fetch length features for the two questions
        length_features = fetch_length_features_for_two_questions(q1_preprocess, q2_preprocess)
        my_dict.update(length_features)



        # Fetch fuzzy features for the two questions
        fuzzy_features = fetch_fuzzy_features_for_two_questions(q1_preprocess, q2_preprocess)
        # print(fuzzy_features)
        my_dict.update(fuzzy_features)


    doit(q1,q2)

    import joblib
    import pickle
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    # preprocessing and feature calculation function

    # Load the model and scaler with joblib if it was saved
    vectorizer = joblib.load('count_vectorizer1.pkl')
    model = joblib.load('XG.pkl')


    # Use the CountVectorizer to transform the text of the new questions
    q1_vectorized = vectorizer.transform([q1]).toarray()
    q2_vectorized = vectorizer.transform([q2]).toarray()


    # Combine the vectorized features (just once)
    combined_vectorized_features = np.concatenate((q1_vectorized, q2_vectorized), axis=1)

    # Convert dictionary to numpy array and reshape
    manually_features = np.array(list(my_dict.values())).reshape(1, -1)

    # Scale the manually calculated features using the loaded scaler
    # scaled_features = scaler.transform(manually_features)

    # Combine both sets of features in the correct order
    final_features = np.concatenate((combined_vectorized_features,manually_features), axis=1)

    # Pass the final features to the model for prediction
    probabilities = model.predict_proba(final_features)

    # # Example probabilities list [x, y]
    # probabilities = [0.7, 0.3]  # You can change this to test with different values

    # Extract values of x and y
    x = probabilities[0][0]
    y = probabilities[0][1]
    # Apply the condition
    if x > 0.5:
        predicted_class = 0  # If x > 0.5, return 0
    elif y > 0.5:
        predicted_class = 1  # If y > 0.5, return 1
   
    
    return predicted_class

def predict_LSTM(q1,q2):

    my_dict = {}
    def preprocess(q):

        q=str(q).lower().strip()
        q=BeautifulSoup(q)
        q=q.get_text()

        q=q.replace("%"," percent")
        q=q.replace("$"," dollar ")
        q=q.replace("₹"," ruppees ")
        q=q.replace("€"," euro ")
        q=q.replace("@"," at ")
        q=q.replace("&"," and ")

        q=q.replace("[math]","")

        q=q.replace(",000,000,000","b")
        q=q.replace(",000,000","m")
        q=q.replace(",000","k")
        q=re.sub(r"([0-9]+)000000000",r"\1b",q)
        q=re.sub(r"([0-9]+)000000",r"\1m",q)
        q=re.sub(r"([0-9]+)000",r"\1k",q)
        contractions= {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it had",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there had",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'alls": "you alls",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had",
        "you'd've": "you would have",
        "you'll": "you you will",
        "you'll've": "you you will have",
        "you're": "you are",
        "you've": "you have"
    }
        q_decontracted=[]

        for word in q.split():
            if word in contractions:

                word=contractions[word]

        q_decontracted.append(word)


        q=" ".join(q_decontracted)
        q=str(q).lower().strip()
        q=q.replace("'ve"," have")
        q=q.replace("n't"," not")
        q=q.replace("'re"," are")
        q=q.replace("'ll"," will")


        pattern=re.compile("\W")
        q=re.sub(pattern," ",q).strip()

        return q

        
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords


    def fetch_token_features_for_two_questions(q1, q2):
        STOP_WORDS = stopwords.words("english")
        SAFE_DIV = 0.0001

        # Initialize token features as a dictionary
        token_features = {}

        # Tokenize the questions
        q1_tokens = q1.split(" ")
        q2_tokens = q2.split(" ")

        # Check for empty tokens
        if len(q1_tokens) == 0 or len(q2_tokens) == 0:
            return token_features

        # Unique non-stop words
        q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
        q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

        # Unique stop words
        q1_stop = set([word for word in q1_tokens if word in STOP_WORDS])
        q2_stop = set([word for word in q2_tokens if word in STOP_WORDS])

        # Calculate common word counts
        common_word_count = len(q1_words.intersection(q2_words))
        common_stop_count = len(q1_stop.intersection(q2_stop))
        common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

        # Calculate features
        token_features["cwc_min"] = common_word_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features["cwc_max"] = common_word_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features["csc_min"] = common_stop_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features["csc_max"] = common_stop_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features["ctc_min"] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features["ctc_max"] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features["last_word_eq"] = int(q1_tokens[-1] == q2_tokens[-1])  # Check if last words are the same
        token_features["first_word_eq"] = int(q1_tokens[0] == q2_tokens[0])    # Check if first words are the same

        return token_features

    import distance

    def fetch_length_features_for_two_questions(q1, q2):
        # Tokenize the input strings
        q1_tokens = q1.split(" ")
        q2_tokens = q2.split(" ")

        # Initialize a dictionary to store the length features
        length_features = {}

        # Return length features if any string is empty
        if len(q1_tokens) == 0 or len(q2_tokens) == 0:
            return length_features

        # Absolute length difference
        length_features["abs_len_diff"] = abs(len(q1_tokens) - len(q2_tokens))

        # Average length
        length_features["mean_len"] = (len(q1_tokens) + len(q2_tokens)) / 2

        # Longest common substring ratio
        strs = list(distance.lcsubstrings(q1, q2))
        try:
            length_features["longest_substr_ratio"] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
        except:
            length_features["longest_substr_ratio"] = 0

        return length_features

    from fuzzywuzzy import fuzz



    def fetch_fuzzy_features_for_two_questions(q1, q2):
        # Initialize a dictionary to store fuzzy features
        fuzzy_features = {}

        # Calculate fuzzy matching features and store them in the dictionary
        fuzzy_features["fuzz_ratio"] = fuzz.QRatio(q1, q2)  # Full ratio
        fuzzy_features["fuzz_partial_ratio"] = fuzz.partial_ratio(q1, q2)  # Partial ratio
        fuzzy_features["token_sort_ratio"] = fuzz.token_sort_ratio(q1, q2)  # Token sort ratio
        fuzzy_features["token_set_ratio"] = fuzz.token_set_ratio(q1, q2)  # Token set ratio

        return fuzzy_features

    def totalword(question1, question2):
        w1 = set(map(lambda word: word.lower().strip(), question1.split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), question2.split(" ")))
        return len(w1) + len(w2)

    def doit(q1,q2):
        q1_preprocess = preprocess(q1)
        q2_preprocess = preprocess(q2)
        # print(q1_preprocess)
        # print(q2_preprocess)



        q1_len = len(q1_preprocess)
        q2_len = len(q2_preprocess)

        my_dict["q1_len"] = q1_len
        my_dict["q2_len"] = q2_len



        q1_num_of_words = len(q1_preprocess.split())
        q2_num_of_words = len(q2_preprocess.split())

        my_dict["q1_num_of_words"] = q1_num_of_words
        my_dict["q2_num_of_words"] = q2_num_of_words




        q1_words = set(q1_preprocess.lower().split())
        q2_words = set(q2_preprocess.lower().split())

        # Find the intersection of the two sets
        common_words = len(q1_words.intersection(q2_words))

        my_dict["common_words"] = common_words




        total_words = totalword(q1_preprocess, q2_preprocess)
        my_dict["total_words"] = total_words



        word_share = round((common_words / total_words) * 100, 2)
        my_dict["word_share"] = word_share



        features = fetch_token_features_for_two_questions(q1_preprocess, q2_preprocess)
        # print(features)
        my_dict.update(features)

    # Fetch length features for the two questions
        length_features = fetch_length_features_for_two_questions(q1_preprocess, q2_preprocess)
        my_dict.update(length_features)



        # Fetch fuzzy features for the two questions
        fuzzy_features = fetch_fuzzy_features_for_two_questions(q1_preprocess, q2_preprocess)
        # print(fuzzy_features)
        my_dict.update(fuzzy_features)

    doit(q1,q2)

    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import numpy as np
    import pickle

    # Load the saved model
    model = tf.keras.models.load_model('LSTM3')

    # Load the saved tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Load the saved vectorizer (assuming it's a count vectorizer or similar)
    with open('count_vectorizer1.pkl', 'rb') as f:
        vectorizer = pickle.load(f)



    # Tokenize and convert questions to sequences
    q1_seq = tokenizer.texts_to_sequences([q1])
    q2_seq = tokenizer.texts_to_sequences([q2])

    # Determine maximum sequence length based on model’s input shape (half of the combined sequence length)
    max_seq_length = model.input_shape[0][1] // 2
    padded_q1_seq = pad_sequences(q1_seq, maxlen=max_seq_length)
    padded_q2_seq = pad_sequences(q2_seq, maxlen=max_seq_length)

    # Combine the padded sequences of both questions
    padded_combined_seq = np.hstack((padded_q1_seq, padded_q2_seq))

    # Use the CountVectorizer to transform the text of the new questions
    q1_vectorized = vectorizer.transform([q1]).toarray()
    q2_vectorized = vectorizer.transform([q2]).toarray()


    # Combine the vectorized features (just once)
    combined_vectorized_features = np.concatenate((q1_vectorized, q2_vectorized), axis=1)

    # Convert the dictionary to an array (ensure order matches model expectations)
    manual_features_array = np.array(list(my_dict.values())).reshape(1, -1)

    # Ensure all inputs are arrays with the correct dimensions
    input_sequence = np.array([padded_combined_seq])  # Shape: (1, sequence_length)
    combined_vector_features = np.hstack((combined_vectorized_features, manual_features_array))  # Shape: (1, vector_feature_count + manual_feature_count)

    # Make sure input_sequence has shape (1, 474) instead of (1, 1, 474)
    input_sequence = padded_combined_seq  # Shape should be (1, 474) now

    # Concatenate vectorized and manually calculated features
    combined_vector_features = np.hstack((combined_vectorized_features, manual_features_array))  # Shape: (1, vector_feature_count + manual_feature_count)

    # Ensure combined_vector_features is also correctly shaped (1, feature_length)
    combined_vector_features = combined_vector_features.reshape(1, -1)

    # Make prediction
    prediction = model.predict([input_sequence, combined_vector_features])

    # Display the prediction result
    # Convert predictions to binary output (1 if > 0.5, else 0)
    binary_prediction = (prediction > 0.5).astype(int)

    x = binary_prediction[0][0]
    if x>0.5:
        return 1
    elif x<0.5:
        return 0

def main():
    st.title("Question Similarity Prediction")

    # Model selection dropdown
    model_choice = st.selectbox("Choose a model:", ["RandomForest", "XGBoost","LSTM"])

    # Input fields for questions
    question1 = st.text_input("Enter the first question:")
    question2 = st.text_input("Enter the second question:")

    # Predict button
    if st.button("Predict"):
        if question1 and question2:
            # Call the selected prediction function based on model_choice
            if model_choice == "RandomForest":
                result = predict_RF(question1, question2)
            elif model_choice == "XGBoost":
                result = predict_XG(question1, question2)
            elif model_choice == "LSTM":
                result = predict_LSTM(question1,question2)
            else:
                result = None
            
            # Display the result based on printed output (0 or 1)
            if result == 1:
                st.write("The questions are Similar.")
            elif result == 0:
                st.write("The questions are Not Similar.")
            else:
                st.write("Unexpected result.")
        else:
            st.write("Please enter both questions.")


if __name__ == "__main__":
    main()
