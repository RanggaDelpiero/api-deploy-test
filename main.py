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

cred_json = {
  "type": "service_account",
  "project_id": "fundup-387016",
  "private_key_id": "a67c3cae82bddaa3b37aaf9f926184c06efd267b",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCwpx7o9BdA7rMJ\nqpyqI4lPM/Slvj0FZwKUpF2Y9oNhxNjFECqp/U32E+ZWDYXhMGyqP6/ww4mZfK1T\ndHJTmQf2+pMQdzeLfylT+M9+jZj7QS9iErYI1BupA5BXpajxvWFG2X+L8Lp8Vxn9\ncb+8pzLHsH8+ty4WVC32YeszlqZSfwXeYEFzeiGRQBJVmaaBtstzCLzcjQ5l1nZi\nk3VD9OsEIKU3hXkNPuxiCUQyvuqdXlhKAmfZPQpw4jmU4ECQO2mpCV+lFkLtJstc\nelurpTp6h2Dm6RxLfsGKcEyW8SEdRQNeyLOyV5gCpM+1OrRvbyy1gF3ycB3u9kD8\nW1anniuFAgMBAAECggEAR2FIJqZW7RhmxN2pT0Brv9LFJOHhg1jT3J8r6N6XSP7C\n/qHhM24Uvf3dgWEWe19XUVXJsJY6eAg+ey3e8nOwGba3jRw3GAlqeDFeGot5yPDW\nhiD8aEXY5Wr4vMnGIeQ9teS12qSLnimN6XC4orDG3pStXfijyUb7iYaYhPB3RXa9\nGt3cLTc0+JfH17DqAwchxhRemQDZ4Wi7fjL2Arp6AGIcouMA6TokXMackZVe7h32\n4LHy+VR8+gHC4TqdeLxb1yvnLPf8ELCaf2EbnRCUvWGs079/dZzboE1VnkNUPmDa\nOUpEhGb/16w3kfDPISN4QNA18NYn2c9KcjYOBQe20wKBgQDgcPUx7GnAADkM+A14\nb0esBgZVLIykG+FnIgV5ckdfZVjWali/AG5q8jcA5Dx+/ktwQH86cOsHIyeGdbEh\nB4xSr5q5QLxLNkPQ5tQSqW2sFJtwZbQ3IatGfVsLb5iQYESja/WmVlWMDTjM5zbT\nzciC6MQ33n0VEbDIVXvc2itl+wKBgQDJffWAzmYaQQoIa6ZBTOxzBS7eBOnPSxjO\nAme1oQmzK3hM48ojdodgtaR07RgrO82bjxlpnJYHIITISvxiIdeEVVrFYFo/Zje5\nMS+aGYSK05TdmzGMX+Vmtb2tCu9OCHoD6AztIeTXB41ST3z2XlVB6MPgCgUxSG4a\nxjzpQkN8fwKBgBjuc05IZLbfT3cRVu257sw9Hxb3C+hu8Gr0bIdBGoyORYAL8C/H\nbHyUy2dd8xpoRRkDER78zB7O2OUmzbZNkFjfCODrP/9a182s1oH8MCKdZ2bk5U/6\nfXwnEKYEj336M6WzqGYB0R7tmRGp3X1JrqxcDu/l1x8wB+M5G7k8wvVhAoGBAIP/\nHTZ9gAvQ8bakduyubPPIwHQ3ucfPxXcnwjMNRSJ35r5QN5rVykgDlrH2pG+mJMK0\nkwxJxUrz9aiU3xOWYe5SUD2fKmAAIZ8TZsDH2LltdEdcpK/2Hn0TsCdNU4nGKdCn\nUtiB7L0lOGJkqlNnZujfiHobdl1buq2Vkk+o1jcXAoGAGH8gZCVj/+zghtn9NFK8\nyZXb9lYKXJegVPeKpnwrvu8U8LmuFsKwcIgvA1beHV8IpXEQJ3toeZsU6BC+std1\nADq7NaTKywd6aREV9bwMc6nbKGvf8zeX/g17d3eADBIPcXs/rvJEVlLnjvYyNT9d\nTnL/hKlL9q9Tb87Yxctl0Js=\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-7vr3t@fundup-387016.iam.gserviceaccount.com",
  "client_id": "104989193780059513595",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-7vr3t%40fundup-387016.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

#cred = credentials.Certificate(app.config['KEY_JSON'])  # Replace with your own service account key file path
cred = credentials.Certificate(cred_json)

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
    app.run(port=3000)
