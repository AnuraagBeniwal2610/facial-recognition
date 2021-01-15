from flask import Flask
from flask import request, jsonify

app = Flask(__name__)

@app.route('/') 
def do():
    return "bhendi"

app.run()
