
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from flask import Flask, render_template, request, jsonify
import re
import pickle

app = Flask(__name__)

count=0

with open('dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)

filename = 'SMS_Fraud_Detection_Model_updated.sav'

classifier = joblib.load(filename)
print("\n Thanks for your patience. \n")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
 global count,classifier
 name = request.form['name']

 if name:
         #var = request.args.get("model_input")
         count_vect = CountVectorizer(analyzer='word')
         var = [name]
         count_vect.fit(dataset['body'])
         var_count = count_vect.transform(var)
         predicted_label = classifier.predict(var_count)
         print("\n This message is ", predicted_label[0])
         return jsonify({'name': 'This message is '+predicted_label[0]})
 return jsonify({'error' : 'Message Field Empty'})

if __name__ == '__main__':

 app.run(debug=True)