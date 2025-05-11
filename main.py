import streamlit as st
import pandas as pd
import cv2
from ultralytics import YOLO
from moviepy.editor import *
from fer import FER
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ---------------------- Load and Preprocess Dataset ----------------------
@st.cache_data
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def normalize_features(df, feature_columns):
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df

# ------------------------ Actor-Critic Models ------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)

        self.action_dim = action_dim

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(state)
        return torch.argmax(probs).item()

    def update(self, state, action, reward):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_one_hot = torch.zeros((1, self.action_dim))
        action_one_hot[0, action] = 1
        reward = torch.tensor([[reward]], dtype=torch.float32)

        q_value = self.critic(state, action_one_hot)
        critic_loss = (q_value - reward).pow(2).mean()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

# ------------------------ YOLO Mood Detection ------------------------
def detect_mood_yolo(frame, model_path='C:/Users/haris/DDGP_music-_recommendation/best_(21).pt'):
    model = YOLO(model_path)
    results = model.predict(source=frame, save=False, imgsz=224, conf=0.3)

    for r in results:
        if r.names and r.boxes:
            label_index = int(r.boxes.cls[0].item())
            label_name = r.names[label_index]
            return label_name
    return "Neutral"


def detect_mood_fer(frame):
    detector = FER(mtcnn=True)
    emotions = detector.detect_emotions(frame)
    if emotions and "emotions" in emotions[0]:
        top_emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
        return top_emotion.lower()
    return "neutral"

# ------------------------ Song Recommendation ------------------------
def recommend_songs(mood, df, agent, disliked, top_n=10):
    mood_mapping = {
        "happy": "Happy", "sad": "Sad", "angry": "Energetic",
        "disgust": "Calm", "fear": "Calm", "neutral": "Calm", "surprise": "Energetic"
    }
    mood = mood_mapping.get(mood.lower(), "Calm")
    mood_songs = df[df['mood'] == mood]
    if mood_songs.empty:
        return pd.DataFrame([{"name": "No songs found", "artist": ""}])

    features = ['danceability', 'acousticness', 'energy', 'valence', 'tempo']
    state = mood_songs[features].mean().values
    action_index = agent.select_action(state)

    recommended = mood_songs[~mood_songs['name'].isin(disliked)][['name', 'artist']]

    recommended = recommended.sample(n=min(top_n, len(recommended)), random_state=42)
    return recommended.reset_index(drop=True), state, action_index

# ------------------------ Streamlit App ------------------------
st.set_page_config(layout="wide")
st.title("🎵 AI Mood-based Song Recommender with Feedback Learning")

# Load data
dataset_path = 'C:/Users/haris/DDGP_music-_recommendation/data_moods.csv'
df = load_dataset(dataset_path)
features = ['danceability', 'acousticness', 'energy', 'valence', 'tempo']
df = normalize_features(df, features)
agent = DDPGAgent(state_dim=5, action_dim=10)

if 'disliked_songs' not in st.session_state:
    st.session_state.disliked_songs = set()
current_state = None
current_action = None

# UI
col1, col2 = st.columns([2, 1])
camera_placeholder = col1.empty()
mood_result = col2.empty()
songs_placeholder = col2.empty()

if "recommend" not in st.session_state:
    st.session_state.recommend = True

if st.button("📸 Capture Mood") or st.session_state.recommend:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camera_placeholder.image(frame_rgb, channels="RGB", caption="Captured Frame")
        mood = detect_mood_fer(frame_rgb)
        mood_result.markdown(f"### Mood Detected: `{mood}`")

        songs, state, action = recommend_songs(mood, df, agent, st.session_state.disliked_songs)
        current_state = state
        current_action = action

        st.session_state.songs = songs
        st.session_state.state = state
        st.session_state.action = action
        st.session_state.recommend = False

if "songs" in st.session_state:
    st.subheader("🎧 Songs for your mood")
    for i, row in st.session_state.songs.iterrows():
        col_song, col_feedback = st.columns([4, 1])
        with col_song:
            st.markdown(f"*{row['name']}* by {row['artist']}")
        with col_feedback:
            col_like, col_dislike = st.columns(2)
            with col_like:
                if st.button("👍", key=f"like_{i}"):
                    agent.update(st.session_state.state, st.session_state.action, 1)
                    st.success("Glad you liked it!")
            with col_dislike:
                if st.button("👎", key=f"dislike_{i}"):
                    st.session_state.disliked_songs.add(row['name'])
                    agent.update(st.session_state.state, st.session_state.action, -1)
                    st.session_state.recommend = True
                    st.rerun()
