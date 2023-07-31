import numpy as np
import pandas as pd
import streamlit as st
from process_datasets import *
import plotly.express as px
import datetime, time
import os

st.set_page_config(layout='wide')

# Get a dict of dataset processing functions
dataset_processor = {
    'extrasensory': process_extrasensory,
    'opportunity': process_opportunity,
    'casas':process_casas,
    'aruba':process_aruba,
    'witham':process_witham,

}

# Data loading config in sidebar
dataset_file = st.sidebar.text_input("Dataset Filepath", value='datasets/extrasensory_dataset.csv')
dataset_func = st.sidebar.selectbox("Dataset Type", ["None"] + list(dataset_processor.keys()), index=1)
result_dir = st.sidebar.text_input("Label Output Filepath", value='gt_labels')

# Data import layer
@st.cache
def load_data(dataset_file, dataset_func):
    df_data = dataset_processor[dataset_func](dataset_file)
    activity_list = df_data.activity.unique().tolist()
    df_data['activity'] = df_data['activity'].apply(lambda x: activity_list.index(x))
    # df_data['datetime'] = pd.to_datetime(df_data['timestamp'], unit='s')
    df_data['date'] = df_data['datetime'].dt.date
    df_id_dates = df_data[['id', 'date']].drop_duplicates()
    df_data = df_data[['id', 'date', 'datetime', 'activity']]
    return df_data, df_id_dates, activity_list


# load data
df_data = None
df_id_dates = None
activity_list = None
if (dataset_file is not None) & (dataset_func != 'None'):
    # if df_data is None:
    #     st.write("loading dataset...")
    df_data, df_id_dates, activity_list = load_data(dataset_file, dataset_func)

# st.write(activity_list)
# st.write(df_data.activity.value_counts())
# st.write("Activity List")
# st.write(activity_list)
left_column, right_column = st.columns(2)
with left_column:
    current_id = st.selectbox("Select id", df_id_dates.id.unique().tolist())

with right_column:
    current_id_dates = df_id_dates[df_id_dates.id == current_id].date.unique().tolist()
    current_date = st.selectbox("Select id", current_id_dates)

# single_column = st.columns(1)
df_current_data = df_data[(df_data.id == current_id) & (df_data.date == current_date)]
# st.write(df_current_data.head(40))
plotly_fig = px.scatter(df_current_data, x="datetime", y="activity", width=1300, height=600)
plotly_fig.update_yaxes(tickvals=list(range(len(activity_list))),
                        ticktext=activity_list, showgrid=True)
st.write(plotly_fig)

# labelling the plot
# check is label dict exists
if 'gt_labels' not in st.session_state.keys():
    st.session_state['gt_labels'] = {}
if current_id not in st.session_state['gt_labels'].keys():
    st.session_state['gt_labels'][current_id] = {}
if current_date not in st.session_state['gt_labels'][current_id].keys():
    st.session_state['gt_labels'][current_id][current_date] = []


prev_label_file = f"GT_marking/{result_dir}/{dataset_func}/{current_id}/{current_date}.txt"
if os.path.exists(prev_label_file):
    if len(st.session_state['gt_labels'][current_id][current_date])==0:
        f = open(prev_label_file,'r')
        lines = f.readlines()
        for line in lines[1:]:
            S = line[:-1].split(",")
            st.session_state['gt_labels'][current_id][current_date].append(S)




def add_context_label():
    st.session_state['gt_labels'][current_id][current_date].append((label_start.strftime('%H:%M'), label_end.strftime('%H:%M'), label_context))


def remove_context_label(label_to_remove_index):
    del st.session_state['gt_labels'][current_id][current_date][label_to_remove_index]


left_col_a, left_col_b, center_col, right_col = st.columns(4)
context_list = ['Amusement', 'ComingIn', 'Commuting', 'Exercising', 'GoingOut', 'HavingMeal', 'HouseWork', 'Inactivity',
                'Meal_Preparation', 'None', 'OfficeWork', 'OnAPhoneCall', 'PhoneCall', 'PreparingMeal', 'Relaxing',
                'Sleeping', 'UsingBathroom',"InAMeeting"]
with left_col_a:
    label_start = st.time_input("context start time", value=datetime.time(0, 0))
with left_col_b:
    label_end = st.time_input("context end time", value=datetime.time(0, 0))
with center_col:
    label_context = st.selectbox("context label", context_list)
with right_col:
    st.write("Click to store label")
    add_label = st.button("add_label", on_click=add_context_label)

label_list = st.session_state['gt_labels'][current_id][current_date]

st.write("Labels created")
left_column, right_column = st.columns(2)
for i, labels in enumerate(label_list):
    with left_column:
        st.button(f"{labels[0]}-----to-----{labels[1]} -------------> {labels[2]}",
                  disabled=True)
    with right_column:
        remove_label = st.button("remove label", on_click=remove_context_label,
                                 args=(i,), key=f"remove_{i}")


def submit_labels():
    label_list = st.session_state['gt_labels'][current_id][current_date]
    if len(label_list) == 0:
        st.write("No labels to submit")
        return None
    curr_timestamp = time.time()
    labels_dir = f"GT_marking/{result_dir}/{dataset_func}/{current_id}"
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    with open(f"{labels_dir}/{current_date}.txt", "w") as f:
        f.write("start_time,end_time,context\n")
        f.writelines(
            [f"{label[0]},{label[1]},{label[2]}\n" for label in label_list])
    st.write("Labels submitted successfully...")
    return None


st.write("Submit labels on the disk")
submit_button = st.button("Submit Labels")
if submit_button:
    submit_labels()
