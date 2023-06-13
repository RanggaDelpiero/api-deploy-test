from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore, auth
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
import os 
from functools import wraps
from flask_cors import CORS
import json

# Initialize Flask application
app = Flask(__name__)   

# Mengambil credential
KEY_JSON = os.environ.get('KEY_JSON')

# Konfigurasi credential
app.config['KEY_JSON'] = KEY_JSON

cred = credentials.Certificate(app.config['KEY_JSON'])  # Replace with your own service account key file path

# Initialize Firestore
firebase_admin.initialize_app(cred)
db = firestore.client()

#read key.json
#with open('key.json', 'r') as file:
	#key_json = file.read()

#load json content
#cred_json = json.loads(key_json)

#use cred_json
#cred = credentials.Certificate(cred_json)


@app.route('/get-recommendation', methods=['POST'])
def get_recomendation_for_startup():
    # Get the id_token from the request (assuming it's provided in the request)
    id_token = request.json.get('id_token')

    # Validate and authenticate the id_token (add your authentication logic here)

    # Check if the id_token is valid
    if id_token is None:
        return jsonify({'error': 'Invalid id_token'})

    # Convert id_token to input_id
    input_id = id_token

    # Query investor_matches collection to get the data
    query_ad = db.collection('investor_matches').document(input_id).get()

    # Check if the document exists
    if query_ad.exists:
        data = query_ad.to_dict()
        investor_matches = data.get('investor_matches', [])
        investor_ids_matches = []

        for investor_id in investor_matches:
            investor_ids_matches.append(investor_id)

        investor_loker_data = []

        for investor_id in investor_ids_matches:
            query = db.collection('investor_loker').document(investor_id).get()
            if query.exists:
                data = query.to_dict()
                investor_loker_data.append(data)
            else:
                print(f"No data found for investor ID: {investor_id}")

        # Process the retrieved investor_loker_data array as needed
        result = []
        for data in investor_loker_data:
            nama_lengkap = data['nama_lengkap']
            nik_investor = str(data['nik_investor'])
            email_investor = data['email_investor']
            target_industri = data['target_industri']
            target_perkembangan = data['target_perkembangan']
            # Add the processed data to the result list
            result.append({
                'nama_lengkap': nama_lengkap,
                'nik_investor': nik_investor,
                'email_investor': email_investor,
                'target_industri': target_industri,
                'target_perkembangan': target_perkembangan
            })

        return jsonify(result)
    else:
        query_ad = db.collection('startup_matches').document(input_id).get()
        data = query_ad.to_dict()
        startup_matches = data.get('startup_matches', [])
        startup_ids_matches = []
        
        for startup_id in startup_matches:
            startup_ids_matches.append(startup_id)
        
        startup_data = []

        for startup_id in startup_ids_matches:
            query = db.collection('startup').document(startup_id).get()
            if query.exists:
                data = query.to_dict()
                startup_data.append(data)
            else:
                print(f"No data found for Startup ID: {startup_id}")

            # Process the retrieved investor_loker_data array as needed
        result = []
        for data in startup_data:
            nama_lengkap = data['nama_lengkap']
            nik_startup = str(data['nik_startup'])
            email_startup = data['email_startup']
            industri_startup = data['industri_startup']
            tingkat_perkembangan_perusahaan = data['tingkat_perkembangan_perusahaan']

            #lanjutin sesuai apa aja atribut yg mau ditampilin di homepage
            result.append({
                'nama_lengkap': nama_lengkap,
                'nik_startup': nik_startup,
                'email_startup': email_startup,
                'industri_startup': industri_startup,
                'tingkat_perkembangan_perusahaan': tingkat_perkembangan_perusahaan
            })

        return jsonify(result)
   


if __name__ == '__main__':
    app.run(port=5000)
