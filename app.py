from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)

# Charger le modèle OVR
with open('model/model.pkl', 'rb') as file:
    ovr_model = pickle.load(file)

# Charger le TfidfVectorizer
tfidf = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english')

# Fonctions de nettoyage du texte
def tokenizer_fct(sentence) :
    # print(sentence)
    sentence_clean = sentence.replace('-', ' ').replace('+', ' ').replace('/', ' ').replace('#', ' ')
    word_tokens = word_tokenize(sentence_clean)
    return word_tokens

# Stop words
stop_w = list(set(stopwords.words('english'))) + ['[', ']', ',', '.', ':', '?', '(', ')', "'", '"', '!', ';', '``', "''", '...', '’', '“', '”']

def stop_word_filter_fct(list_words) :
    filtered_w = [w for w in list_words if not w in stop_w]
    filtered_w2 = [w for w in filtered_w if len(w) > 2]
    return filtered_w2

# lower case et alpha
def lower_start_fct(list_words) :
    lw = [w.lower() for w in list_words if (not w.startswith("@")) 
                                       and (not w.startswith("#"))
                                       and (not w.startswith("http"))]
    return lw

# Lemmatizer (base d'un mot)
def lemma_fct(list_words) :
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]
    return lem_w

# Fonction de préparation du texte pour le bag of words avec lemmatization
def transform_bow_lem_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lem_w)
    return transf_desc_text


@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.json['text']
    cleaned_text = transform_bow_lem_fct(input_text)
    text_tfidf = tfidf.transform([cleaned_text])
    prediction_proba = ovr_model.predict_proba(text_tfidf)

    threshold = 0.1
    sorted_indices = np.argsort(-prediction_proba[0])
    result = [(ovr_model.classes_[i], prediction_proba[0][i]) for i in sorted_indices if prediction_proba[0][i] >= threshold]

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
    