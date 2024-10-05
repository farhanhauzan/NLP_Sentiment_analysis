# Library imports 
import pandas as pd
import os
import pickle, re
import numpy as np
import tensorflow as tf
import sqlite3

# Machine Learning library imports
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from werkzeug.utils import secure_filename
from flask import Flask, jsonify,request, make_response, redirect, url_for, flash, render_template
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
from io import StringIO
from pathlib import Path
from keras.models import load_model
from collections import defaultdict

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()





app = Flask(__name__)

# File path directory
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# print("APP_ROOT", APP_ROOT)
# print("UPLOAD_FOLDER", UPLOAD_FOLDER)


class CustomFlaskAppWithEncoder(Flask):
    json_provider_class = LazyJSONEncoder

app = CustomFlaskAppWithEncoder(__name__)



swagger_template = dict(
    info = {
        'title': LazyString(lambda: "Platinum Challange"),
        'version': LazyString(lambda: "1.0.0"),
        'description': LazyString(lambda: "Platinum Challange Documentation"),
    },
    host = LazyString(lambda: request.host)
)


swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "docs",
            "route": "/docs.json",
        }
    ],
    "static_url_path": "/flasgger_static",
    #"static_folder": "static", #must be set by user
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template,config = swagger_config)


#Loading Neural Network pickple file
pickle.load(open("model_of_nn/model-sentiment.p", "rb"))
pickle.load(open("model_of_nn/feature.p", "rb"))


#LSTM

#Load LSTM tokenizer pickle file
tokenizer = pickle.load(open('resource_of_lstm/tokenizer.pickle', 'rb'))

#Load LSTM pickle file for x_pad_sequence
file = open('resource_of_lstm/x_pad_sequences.pickle', 'rb')
feature_file_form_lstm = pickle.load(file)
file.close()

#Load LSTM model
model_file_from_lstm = load_model('model_of_lstm/modelLSTM.h5')



# Cleaning Function for word cleansing
def cleaning(text):
    if isinstance(text, str):
        # lower case text
        text = text.lower()
        # removing whitespaces
        text = text.strip()

        # removing USER tag    
        text = re.sub('user',' ' , text)
        # removing URL tag    
        text = re.sub('url', ' ', text)
        # removing "RT" tag    
        text = re.sub('rt', ' ', text)
        # removing random url
        text = re.sub(r'https?:[^\s]+', ' ', text)      
                        
        # removing tab
        text = re.sub('\t', ' ', text)
        #  removing random /xf character
        text = re.sub('x[a-z0-9]{2}', ' ', text)
        #  removing code "newline"  
        text =  text.replace('\\n', ' ')
        # removing remain symbols  
        text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
        text = re.sub(r"[-()\"#/@;:{}`+=~|.!?,'0-9]", "  ", text)
        # removing remain whitespaces
        text = re.sub(r' \s+', ' ',text)
        # removing whitespaces
        text = text.strip()

        return text
    else:
        return text
    
# Stemming Function for word normalization
def stemming(text):
    text = stemmer.stem(text)
    return text

# Combined Function for word cleaning and stemming
def cleansing(text):
    if isinstance(text, str):
        text = cleaning(text)
        text = stemming(text)
        return text
    elif isinstance(text, pd.Series):
        return text.apply(lambda x: cleansing(x))
    else:
        return text

# Label sentiment
sentiment = ['negative', 'neutral', 'positive']


@app.route('/', methods=['GET'])
def hello_world():
    json_response = {
        'status_code': 200,
        'description': "Platinum Challange",
        'description1': "SILAHKAN TAMBAHKAN /docs untuk masuk ke UI SWEGGER",
        'data': "DATA SCIENCE - WAVE 21",
    }

    response_data = jsonify(json_response)
    return response_data


###########################################LSTM MODEL#######################################

@swag_from("docs/LSTM_text.yml", methods=['POST'])
@app.route('/lstm_text', methods=['POST'])
def lstm_text():

    # Get text
    original_text = request.form.get('text')
    # applying cleansing function
    text = [cleansing(original_text)]
    # applying feature extraction
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_form_lstm.shape[1])
    # Inference
    prediction = model_file_from_lstm.predict(feature)
    # print('list_sentiment :',sentiment)
    # print ('prediction : ', prediction)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    # Define API response
    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using LSTM",
        'data': {
            'text': text[0],
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data


@swag_from("docs/LSTM_file.yml", methods=['POST'])
@app.route('/LSTM-Processing', methods=['POST'])

def upload_file():
    # memulai proses upload file, save and read file 
    if 'upload_file' not in request.files:
        return redirect(request.url)
    
    file = request.files['upload_file']

    if file.filename == '':
        return redirect(request.url)
    
    # Save file to directory
    csv_filename = secure_filename(file.filename)
    file.save(os.path.join(UPLOAD_FOLDER, csv_filename))
    path = os.path.join(UPLOAD_FOLDER)
    print("data_film", csv_filename)

    # Read CSV file
    file_path = Path(path) / csv_filename
    # try:
    read = file_path.read_text(encoding='latin-1')


    # Pandas read csv to dataframe
    df = pd.read_csv(StringIO(read), header=None)

    # return first 10 row on csv
    read_csv = df[0].head(10)
    print("read_csv", read_csv)

    # applying cleansing function
    cleaned_text = []
    for raw_text in read_csv:
        text=(re.sub(r"(USER|(www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))|([^a-zA-Z0-9])"," ", raw_text).upper())
        print("text_clean", text)

        feature = tokenizer.texts_to_sequences(text)
        feature = pad_sequences(feature, maxlen=feature_file_form_lstm.shape[1])
    

        # Inference
        prediction = model_file_from_lstm.predict(feature, batch_size=32)
        get_sentiment = sentiment[np.argmax(prediction[0])]
        print("get_sentiment :", get_sentiment)
        cleaned_text.append({
            # 'raw_text': raw_text,
            'text': text,
            'get_sentiment': get_sentiment
        })


    # Define API response
    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using LSTM",
        # 'data': {
        #     'text': text,
        #     'sentiment': get_sentiment
        # },
        'data': cleaned_text
    }

    # except Exception as read:

    response_data = jsonify(json_response)
    return response_data
    # return read_csv


###########################################NEURAL NETWORK MODEL#######################################

@swag_from("docs/NN_text.yml", methods=['POST'])
@app.route('/nn_text', methods=['POST'])
def nn_text():

    # Get text from form
    text = request.form.get('text')

    # create list from text and applying cleansing funtion as well
    text_list = [cleansing(text)]

    # Load model CountVectorizer
    count_vect = pickle.load(open("model_of_nn/feature.p", "rb"))

    # applying text vectorizer
    text_vect = count_vect.transform(text_list)

    # Load model Neural Network
    model = pickle.load(open("model_of_nn/model-sentiment.p", "rb"))

    # predict text using model MLPClassifier
    result = model.predict(text_vect)

    result = result.tolist()
    

    # Define API response
    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using NN",
        'data': {
            'text': text,
            'sentiment': result
        },
    }
    response_data = jsonify(json_response)
    return response_data

@swag_from("docs/NN_file.yml", methods=['POST'])
@app.route('/nn_file', methods=['POST'])
def nn_file():
    if 'upload_file' not in request.files:
        return redirect(request.url)
    
    file = request.files['upload_file']

    if file.filename == '':
        return redirect(request.url)
    
    # Save file to directory
    csv_filename = secure_filename(file.filename)
    file.save(os.path.join(UPLOAD_FOLDER, csv_filename))
    path = os.path.join(UPLOAD_FOLDER)
    print("data_film", csv_filename)

    # Read CSV file
    file_path = Path(path) / csv_filename
    try:
        read = file_path.read_text(encoding='latin-1')

        # pandas read csv to dataframe
        df = pd.read_csv(StringIO(read), header=None)
        print("df", df)

        # show 10 first row of data
        read_csv = df[0].head(5)

        # applying cleansing function and create text to list
        text_list = read_csv.apply(lambda x: cleansing(x)).tolist()

        # Load model CountVectorizer
        count_vect = pickle.load(open("model_of_nn/feature.p", "rb"))

        # applying text vectorizer
        text_vect = count_vect.transform(text_list)

        # Load Neural Network model
        model = pickle.load(open("model_of_nn/model-sentiment.p", "rb"))

        # predict text using model MLPClassifier
        result = model.predict(text_vect)

        result = result.tolist()



        # Define API response
        json_response = {
            'status_code': 200,
            'description': "Result of Sentiment Analysis using NN",
            'data': [ 
                {
                    'text': text,
                    'sentiment': sentiment
                }
                for text, sentiment in zip(read_csv.tolist(), result)
            ],
        }
        
        response_data = jsonify(json_response)
        return response_data
    
    except Exception as e:
        return jsonify({'error': str(e)})
    


if __name__ == '__main__':
    app.run(debug=True)