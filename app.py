# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# from model import make_model, summary_to_feature
# import os

# app = Flask(__name__)
# CORS(app)

# MODEL_PATH = 'policy.h5'

# # actions: easy, medium, hard, superhard
# actions = ['easy','medium','hard','superhard']

# if os.path.exists(MODEL_PATH):
#     from tensorflow.keras.models import load_model
#     policy = load_model(MODEL_PATH)
# else:
#     policy = make_model()

# # a tiny experience buffer
# buffer = []

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     summary = request.json or {}
#     feat = summary_to_feature(summary)
#     probs = policy.predict(feat.reshape(1,-1), verbose=0)[0]
#     idx = int(np.random.choice(len(probs), p=probs))
#     difficulty = actions[idx]
#     # map difficulty to params
#     mapping = {
#         'easy': { 'rows': 10, 'cols': 16, 'coinCount': 6, 'obstacleDensity': 0.06, 'difficulty':'easy'},
#         'medium': { 'rows': 12, 'cols': 18, 'coinCount': 8, 'obstacleDensity': 0.10, 'difficulty':'medium'},
#         'hard': { 'rows': 14, 'cols': 22, 'coinCount': 12, 'obstacleDensity': 0.16, 'difficulty':'hard'},
#         'superhard': { 'rows': 16, 'cols': 26, 'coinCount': 18, 'obstacleDensity': 0.22, 'difficulty':'superhard'}
#     }
#     return jsonify({ 'params': mapping[difficulty], 'probs': probs.tolist() })

# @app.route('/update', methods=['POST'])
# def update():
#     # receives {userId, levelId, metrics}
#     data = request.json or {}
#     metrics = data.get('metrics', {})
#     summary = { 'lastPlays': [ { 'metrics': metrics } ] }
#     feat = summary_to_feature(summary)
#     # simple reward: finished -> +1, else proportional to coins
#     reward = 1.0 if metrics.get('finished') else (metrics.get('coinsCollected',0)/max(1,metrics.get('totalCoins',1)))
#     # store tiny experience
#     buffer.append((feat, reward))
#     if len(buffer) >= 8:
#         # perform a tiny policy-gradient-like update: create training examples: prefer actions that resulted in higher reward.
#         X = np.stack([b[0] for b in buffer])
#         y = np.zeros((len(X), 4)) + 0.25
#         for i,(f,r) in enumerate(buffer):
#             if r > 0.8: 
#                 y[i,1]=0.05; y[i,2]=0.55; y[i,3]=0.35
#             elif r>0.4: 
#                 y[i,1]=0.55; y[i,2]=0.35; y[i,3]=0.10
#             else: 
#                 y[i,0]=0.7; y[i,1]=0.2
#         policy.fit(X, y, epochs=6, verbose=0)
#         policy.save(MODEL_PATH)
#         buffer.clear()
#     return jsonify({'ok':True})

# if __name__ == '__main__':
#     app.run(port=5001, debug=False)


from flask import Flask, request, jsonify
from flask_cors import CORS
from model import summary_to_feature, probs_from_feature, reward_from_metrics
import random
import os
import json

app = Flask(__name__)
CORS(app)

actions = ['easy', 'medium', 'hard', 'superhard']

# tiny persistent experience file (optional)
STORAGE = 'rl_data.json'
if os.path.exists(STORAGE):
    try:
        with open(STORAGE, 'r') as f:
            saved = json.load(f)
    except Exception:
        saved = {}
else:
    saved = {}

@app.route('/recommend', methods=['POST'])
def recommend():
    summary = request.json or {}
    feat = summary_to_feature(summary)
    probs = probs_from_feature(feat)
    idx = int(random.choices(range(len(actions)), weights=probs, k=1)[0])
    difficulty = actions[idx]
    mapping = {
        'easy': { 'rows': 10, 'cols': 16, 'coinCount': 6, 'obstacleDensity': 0.06, 'difficulty':'easy'},
        'medium': { 'rows': 12, 'cols': 18, 'coinCount': 8, 'obstacleDensity': 0.10, 'difficulty':'medium'},
        'hard': { 'rows': 14, 'cols': 22, 'coinCount': 12, 'obstacleDensity': 0.16, 'difficulty':'hard'},
        'superhard': { 'rows': 16, 'cols': 26, 'coinCount': 18, 'obstacleDensity': 0.22, 'difficulty':'superhard'}
    }
    return jsonify({ 'params': mapping[difficulty], 'probs': probs })

# keep a tiny buffer for diagnostics
BUFFER = []

@app.route('/update', methods=['POST'])
def update():
    data = request.json or {}
    metrics = data.get('metrics', {})
    r = reward_from_metrics(metrics)
    BUFFER.append(r)
    # Optionally save a log for inspection
    saved.setdefault('history', []).append({'metrics': metrics, 'reward': r})
    try:
        with open(STORAGE, 'w') as f:
            json.dump(saved, f)
    except Exception:
        pass
    # No heavy training; we just accept the metrics
    return jsonify({'ok': True})

if __name__ == '__main__':
    app.run(port=5001, debug=False)
