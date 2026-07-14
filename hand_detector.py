import mediapipe as mp
import streamlit as st

st.write("MediaPipe Version:", mp.__version__)
st.write("MediaPipe:", dir(mp))
st.stop()
