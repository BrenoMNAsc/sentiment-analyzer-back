# Importando as bibliotecas necessárias
from flask import Flask, request
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

# Baixando os pacotes necessários do NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Inicializando o aplicativo Flask
app = Flask(__name__)

# Definindo as funções auxiliares
def clean(text):
    port = PorterStemmer()
    stopword = nltk.corpus.stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    lower = [word.lower() for word in tokens]
    no_stopwords = [word for word in lower if word not in stopword]
    no_alpha = [word for word in no_stopwords if word.isalpha()]
    lemm_text = [port.stem(word) for word in no_alpha]
    clean_text = lemm_text
    return ' '.join(word for word in clean_text)

def extract_postags(text):
    tokens = nltk.word_tokenize(text)
    return [tag for word, tag in nltk.pos_tag(tokens)]

def join_postags(postags):
    return ' '.join(postags)

# Definindo a rota de previsão
@app.route('/predict', methods=['GET'])
def classify_text():
    args = request.args
    text = args.get('text')
    clean_text = clean(text)
    postags = extract_postags(clean_text)
    postags_str = join_postags(postags)
    text_postags = clean_text + ' ' + postags_str
    X_final = tfid.transform([text_postags])
    prediction = random_forest_classifier.predict(X_final)
    return f'Sentiment: {prediction}'

# Função para treinar o modelo
def train_model():
    global tfid, random_forest_classifier
    # Carregando os dados
    df = pd.read_csv('./content/data.csv')

    # Pré-processando os dados
    clean_df = df
    for i in range(len(df['Sentence'])):
        clean_df['Sentence'][i] = clean(df['Sentence'][i])

    clean_df['postags'] = clean_df['Sentence'].apply(extract_postags)
    clean_df['postags_str'] = clean_df['postags'].apply(join_postags)
    clean_df['text_postags'] = clean_df['Sentence'] + ' ' + clean_df['postags_str']
    clean_df['Sentiment'] = clean_df['Sentiment'].replace({'neutral': 0, 'positive': 1, 'negative': -1})
    y = clean_df['Sentiment']

    # Transformando os dados
    tfid = TfidfVectorizer()
    X_final = tfid.fit_transform(clean_df['text_postags'])

    # Balanceando os dados
    smote = SMOTE()
    x_sm, y_sm = smote.fit_resample(X_final, y)

    # Dividindo os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(x_sm, y_sm, test_size=0.1, random_state=3)

    # Treinando o modelo
    random_forest_classifier = RandomForestClassifier()
    random_forest_classifier.fit(X_train, y_train.values.ravel())

# Treinando o modelo
train_model()

# Executando o aplicativo Flask
if __name__ == '__main__':
    app.run()
