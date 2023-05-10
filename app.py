from flask import Flask, request, current_app, jsonify
import pickle
import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Créer une classe personnalisée pour gérer la sérialisation des objets numpy
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        else:
            return super(CustomJSONEncoder, self).default(obj)

# Créer une application Flask
app = Flask(__name__)
app.json_encoder = CustomJSONEncoder


# Charger le modèle OVR
with open('model/model.pkl', 'rb') as file:
    ovr_model = pickle.load(file)

# Charger le TfidfVectorizer ajusté
with open('model/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

#Charger le mlb ajusté
with open('model/mlb.pkl', 'rb') as f:
    mlb = pickle.load(f)


    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

class_mapping = {i: class_name for i, class_name in enumerate(mlb.classes_)}

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

# Fonction personnalisée pour jsonify
def custom_jsonify(*args, **kwargs):
    return current_app.response_class(
        json.dumps(dict(*args, **kwargs), cls=NumpyEncoder),
        mimetype=current_app.config['JSONIFY_MIMETYPE']
    )

@app.route('/')
def home():
    return "API Flask pour la prédiction de texte"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_text = data['text']
    
    cleaned_text = transform_bow_lem_fct(input_text)
    text_tfidf = tfidf.transform([cleaned_text])
    prediction_proba = ovr_model.predict_proba(text_tfidf)

    threshold = 0.1
    sorted_indices = np.argsort(-prediction_proba[0])
    result = [(class_mapping[ovr_model.classes_[i]], prediction_proba[0][i]) for i in sorted_indices if prediction_proba[0][i] >= threshold]

    return jsonify(result)




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)