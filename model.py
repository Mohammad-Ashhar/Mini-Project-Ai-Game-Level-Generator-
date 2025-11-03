# # Very small policy network that maps a user summary vector to a distribution over 4 difficulty choices.
# import numpy as np
# from tensorflow import keras
# from tensorflow.keras import layers

# def make_model(input_dim=10, n_actions=4):
#     model = keras.Sequential([
#         layers.Input(shape=(input_dim,)),
#         layers.Dense(32, activation='relu'),
#         layers.Dense(32, activation='relu'),
#         layers.Dense(n_actions, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy')
#     return model

# # utilities

# def summary_to_feature(summary, max_len=10):
#     # summary is a dict possibly containing lastPlays: [{metrics}]
#     last = summary.get('lastPlays', [])[-max_len:]
#     feat = []
#     for p in last:
#         m = p.get('metrics', {})
#         feat += [1.0 if m.get('finished') else 0.0, m.get('coinsCollected',0)/max(1,m.get('totalCoins',1)), min(1.0, 1.0/(1.0+m.get('timeTaken',0)) )]
#     # pad to input size
#     feat = feat + [0.0]* (max(0, 10 - len(feat)))
#     return np.array(feat[:10])

# Lightweight model utilities (no TensorFlow)
import math

def summary_to_feature(summary, max_len=10):
    last = summary.get('lastPlays', [])[-max_len:]
    feat = []
    for p in last:
        m = p.get('metrics', {})
        feat += [1.0 if m.get('finished') else 0.0,
                 (m.get('coinsCollected', 0) / max(1, m.get('totalCoins', 1))),
                 min(1.0, 1.0 / (1.0 + m.get('timeTaken', 0)))]
    feat = feat + [0.0] * (max(0, 10 - len(feat)))
    return feat[:10]

# A very small "policy" stored in-memory as weights for difficulties
# difficulties = ['easy','medium','hard','superhard']
DEFAULT_PROBS = [0.5, 0.3, 0.15, 0.05]

# A simple updater that adjusts probabilities based on reward
def probs_from_feature(feat):
    # feat contains recent play outcomes; produce a score in [0,1]
    if not feat:
        return DEFAULT_PROBS[:]
    # compute average finished rate (feat positions 0,3,6...), coins ratio (1,4,7...), speed proxy (2,5,8...)
    n = len(feat) // 3
    if n == 0:
        return DEFAULT_PROBS[:]
    finished = 0.0; coins = 0.0; speed = 0.0
    for i in range(n):
        finished += feat[i*3 + 0]
        coins += feat[i*3 + 1]
        speed += feat[i*3 + 2]
    finished /= n; coins /= n; speed /= n
    # Heuristic: higher finished/colection => increase difficulty
    score = 0.5*finished + 0.3*coins + 0.2*speed  # range ~0..1
    # Map score to distribution
    if score < 0.3:
        return [0.8, 0.15, 0.04, 0.01]
    elif score < 0.6:
        return [0.4, 0.4, 0.15, 0.05]
    elif score < 0.85:
        return [0.15, 0.5, 0.25, 0.10]
    else:
        return [0.05, 0.25, 0.45, 0.25]

def reward_from_metrics(metrics):
    if not metrics:
        return 0.0
    return 1.0 if metrics.get('finished') else (metrics.get('coinsCollected',0) / max(1, metrics.get('totalCoins',1)))
