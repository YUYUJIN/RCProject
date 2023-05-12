from flask import Flask, render_template, Response,jsonify,request,send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

