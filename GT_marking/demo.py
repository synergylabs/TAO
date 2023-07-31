import numpy as np
import pandas as pd
import streamlit as st
import requests
import json
from process_datasets import *
import plotly.express as px
import datetime, time
import os

# url = st.text_input("Context Sensing URL", value="https://api.tumblr.com/v2/blog/puppygifs.tumblr.com/posts/photo?api_key=YOUR_KEY_HERE")
url = "https://api.tumblr.com/v2/blog/puppygifs.tumblr.com/posts/photo?api_key=YOUR_KEY_HERE"

dummy_sample = [
    {
        "color": "red",
        "value": "#f00"
    },
    {
        "color": "green",
        "value": "#0f0"
    },
    {
        "color": "blue",
        "value": "#00f"
    },
    {
        "color": "cyan",
        "value": "#0ff"
    },
    {
        "color": "magenta",
        "value": "#f0f"
    },
    {
        "color": "yellow",
        "value": "#ff0"
    },
    {
        "color": "black",
        "value": "#000"
    }
]

with st.empty():
    for seconds in range(60):
        # response = requests.get(url)
        # response_dict = json.loads(response.text)
        st.write(f"⏳ {seconds} seconds have passed")
        st.write(dummy_sample[seconds%len(dummy_sample)])
        time.sleep(1)
    st.write("✔️ 1 minute over!")
