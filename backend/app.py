import os
from flask import Flask, request, jsonify
from audio_analyzer import analyze_audio

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    audio_file = request.files['file']
    result = analyze_audio(audio_file)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
