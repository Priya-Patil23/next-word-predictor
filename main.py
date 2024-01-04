from flask import Flask, render_template , request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
import pickle

wordnet = WordNetLemmatizer()
model = load_model('nextwordprediction_oncustomdata.h5')
tokenizer = pickle.load(open("tokenizer" , "rb"))

contractions = {
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
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

def predicttion(text , number):
    string = text
    for i in range(number):
        tokened = tokenizer.texts_to_sequences([string])
        padded = pad_sequences(tokened , maxlen = 51 , padding = 'pre')
        idx = np.argmax(model.predict(padded))
        for word, index in tokenizer.word_index.items():
            if index == idx:
                string = string + " " + word
    return string


def clean_text(x):
    x = x.lower()
    x = re.sub(r'https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE)
    x = re.sub(r'\<a href', ' ', x)
    x = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', x)
    x = re.sub(r'\'', ' ', x)
    x = re.sub(r"'s\b", "", x)
    x = re.sub("[^a-zA-Z]", " ", x)
    x = ' '.join([word for word in x.split() if len(word) >= 3])

    text_contract = []
    lemm_text = []
    for i in x.split(" "):
        if i in contractions:
            text_contract.append(contractions[i])
        else:
            text_contract.append(i)
    for i in text_contract:
        lemm_text.append(wordnet.lemmatize(i))

    return " ".join(lemm_text)


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict' ,methods = ['POST'])
def predict():
    text = request.form.get('text')
    number = request.form.get('number')
    cleaned_text = clean_text(text)
    result = predicttion(cleaned_text , int(number))
    return render_template('index.html' ,result = result)

if __name__ == '__main__':
    app.run(debug=True)
