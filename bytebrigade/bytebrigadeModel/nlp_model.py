# # import joblib

# # # Load your trained NLP model (for example, a scikit-learn model)
# # model = joblib.load("bytebrigadeModel/model2/3_class_tfidfNeural.h5")

# Example function to classify text
# def classify_complaint(complaint_text):
    # Process text (e.g., preprocessing, tokenization, etc. depending on your model)
    # This is just a placeholder; you need to adjust it to your model's requirements
    #processed_text = preprocess_text(complaint_text) 
    
    # Predict category
    #category = model.predict([processed_text])  # Assuming the model takes a list of sentences
    #return category[0]
    # return "spam"

# def preprocess_text(text):
    # Add any preprocessing steps here
    # For example: Lowercasing, removing stop words, etc.
    # return text.lower()

from tensorflow import keras
import joblib
import numpy as np
import emoji
import re
import string
from bs4 import BeautifulSoup
import os


weight = 1.1
predictions = ""
actual_ls = ["Women/Child Related Crime","Financial Fraud Crimes","Other Cyber Crime"]

def class_label(x):
    return actual_ls[x]

def category_classification(complaint):
    print("Complain text :", complaint)
    print("Current working directory : ", os.getcwd())

    model = keras.models.load_model('model/3_class_tfidfNeural.h5')
    tfidf_vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
    selector = joblib.load('model/selector.pkl')
    svd = joblib.load('model/svd.pkl')

    new_sequences = tfidf_vectorizer.transform([complaint])
    new_sequences_selected = selector.transform(new_sequences)
    new_sequences_reduced = svd.transform(new_sequences_selected)
    new_sequences_reduced = np.expand_dims(new_sequences_reduced, axis=2)
    predictions = model.predict(new_sequences_reduced)

    return predictions

def clean_text(text):
    text = str(text)
    '''Clean emoji, Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = emoji.demojize(text)
    text = re.sub(r'\:(.*?)\:','',text)
    text = str(text).lower()    #Making Text Lowercase
    text = re.sub('\[.*?\]', '', text)
    #The next 2 lines remove html text
    text = BeautifulSoup(text, 'lxml').get_text()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "'")
    text = re.sub(r"[^a-zA-Z?.!,¿']+", " ", text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # eg: "he is a boy." => "he is a boy ."
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    text = text.strip()
    text = text.split()
    return " ".join(text)


def sub_category_classification(complaint_text, predictions):
    # for i in range(0,len(complaint_text)):
    #     complaint_text[i] = clean_text(complaint_text[i])
    complaint_text = [clean_text(text) for text in complaint_text]

    tfidf_vectorizer = joblib.load('model2/tfidf_vectorizer.pkl')
    selector = joblib.load('model2/selector.pkl')
    scaler = joblib.load('model2/scaler.pkl')
    svd = joblib.load('model2/svd.pkl')
    model = keras.models.load_model('model2/3_class_tfidfNeural.h5')

    # TF-IDF Vectorizer
    new_tfidf = tfidf_vectorizer.transform(complaint_text)
    # Feature Selector
    new_selected = selector.transform(new_tfidf)
    # StandardScaler
    new_scaled = scaler.transform(new_selected)
    # SVD
    new_reduced = svd.transform(new_scaled)
    # Neural Network input reshape
    new_input = np.expand_dims(new_reduced, axis=2)
    # Prediction
    sec_predictions = model.predict(new_input)

    predicted = (predictions[0]*weight)+sec_predictions[0]

    predicted_class_index = int(np.where(predicted == np.max(predicted))[0])
    result = class_label(predicted_class_index)

    return result

