import numpy as np
import pandas as pd
import streamlit as st
import requests
import datetime, time
import os

st.set_page_config(layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:50px !important;
}

.medium-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)


from owlready2 import *
import rdflib

models_data_dir = '../activity_ml_models'
model_name = 'RandomForestClassifier'
datawrite_dir = '../activity_ml_models/testdata.csv'

# onto = get_ontology("ontological_models/context-v2.owl").load()
g = rdflib.Graph()
g.parse("../ontological_models/context-v2.n3")
g.serialize("../ontological_models/context-v2.ttl")

classes = ['Walking', 'Eating', 'DescendingStairs', 'Sweeping',
           'Hiking', 'Cycling', 'LyingDown', 'Jogging', 'Sleeping',
           'Running', 'Sitting']

activity_rename_mapping = {
    "Lying down": 'LyingDown',
    "Sitting": 'Sitting',
    "Walking": 'Walking',
    "Running": 'Running',
    "Bicycling": 'Cycling',
    "Sleeping": 'Sleeping',
    "Lab work": 'SittingOffice',
    "In class": 'SittingOffice',
    "In a meeting": 'Meeting',
    "Drive - I'm the driver": 'Driving',
    "Drive - I'm a passenger": 'Driving',
    "Exercise": 'Hiking',
    "Cooking": 'Cooking',
    "Shopping": 'Shopping',
    "Strolling": 'Running',
    "Drinking (alcohol)": 'Drinking',
    "Bathing - shower": 'Shower',
    "Cleaning": 'VacuumHome',
    "Doing laundry": 'VacuumHome',
    "Washing dishes": 'VacuumHome',
    "WatchingTV": 'WatchingTv',
    "Surfing the internet": 'ReadingOffice',
    "Singing": 'Dancing',
    "Talking": 'Talking',
    "Computer work": 'TypingOffice',
    "Eating": 'Eating',
    "Toilet": 'Toilet',
    "Grooming": 'Grooming',
    "Dressing": 'Grooming',
    "Stairs - going up": 'ClimbingStairs',
    "Stairs - going down": 'DescendingStairs',
    "Standing": 'Standing',
    "With co-workers": 'Meeting',
    "With friends": 'Dancing',
    "Step_Out": 'StepOut',
    "Other_Activity": 'None',
    "Toilet": 'Toilet',
    "Phone": 'OnPhone',
    "Personal_Hygiene": 'Grooming',
    "Leave_Home": 'StepOut',
    "Enter_Home": 'StepIn',
    "Relax": 'LyingDown',
    "Sleep_Out_Of_Bed": 'LyingDown',
    "Drink": 'Drinking',
    "Watch_TV": 'WatchingTv',
    "Dress": 'Grooming',
    "Evening_Meds": 'Eating',
    "Wake_Up": 'Sleeping',
    "Read": 'TypingOffice',
    "Morning_Meds": 'Eating',
    "Cook_Breakfast": 'Cooking',
    "Eat_Breakfast": 'Eating',
    "Bathe": 'Shower',
    "Cook_Lunch": 'Cooking',
    "Eat_Lunch": 'Eating',
    "Wash_Lunch_Dishes": 'VacuumHome',
    "Go_To_Sleep": 'Sleeping',
    "Sleep": 'Sleeping',
    "Bed_Toilet_Transition": 'Toilet',
    "Wash_Breakfast_Dishes": 'VacuumHome',
    "Work_At_Table": 'SittingOffice',
    "Groom": 'Grooming',
    "Cook": 'Cooking',
    "Eat": 'Eating',
    "Cook_Dinner": 'Cooking',
    "Eat_Dinner": 'Eating',
    "Wash_Dinner_Dishes": 'VacuumHome',
    "Wash_Dishes": 'VacuumHome',
    "Entertain_Guests": 'Dancing',
    "Take_Medicine": 'Eating',
    "Work": 'SittingOffice',
    "Exercise": 'Running',
    "Work_On_Computer": 'SittingOffice',
    "Nap": 'LyingDown',
    "Work_At_Desk": 'SittingOffice',
    "Laundry": 'VacuumHome',
    "coffeemachine": "Cooking",
    "doorknock": "StepIn",
    "dooropen": "StepIn",
    "eating": "Eating",
    "jumping": "Hiking",
    "mouse": "TypingOffice",
    "phonering": "OnPhone",
    "running": "Hiking",
    "sweeping": "VacuumHome",
    "talking": "Talking",
    "tv": "WatchingTv",
    "typing": "TypingOffice",
    "vacuum": "VacuumHome",
    "writing": "SittingOffice"
}


# Load TAO ontology
load_label_ontology_file = "../ontological_models/ontology_labels_v2.csv"
df_tao_onto = pd.read_csv(load_label_ontology_file, names=['activity', 'context'])

# Load activity labels:

real_world_gt_path = "/Users/sudershan/all-documents/cmu-research/synergylabs/git-projects/context-sensing/" \
                     "context_sensing_v2/data_collection_pipeline/real_gt_data/" \
                     "real_world_ontology_predictions_v1.csv"
real_world_gt_path_mites = "/Users/sudershan/all-documents/cmu-research/synergylabs/git-projects/context-sensing/" \
                           "context_sensing_v2/data_collection_pipeline/real_gt_data/" \
                           "real_world_mites_ontology_predictions_v1.csv"

# Video Annotated Ground Truth
df_real_world_gt_onto = pd.read_csv(real_world_gt_path)
df_real_world_gt_onto = df_real_world_gt_onto[df_real_world_gt_onto['tao_prediction'].notna()]

# Mites Ground Truth
df_real_world_mites_onto = pd.read_csv(real_world_gt_path_mites)
df_real_world_mites_onto = df_real_world_mites_onto[df_real_world_mites_onto['tao_prediction'].notna()]


def return_mapping(key_input):
    if type(key_input) == list:
        res = []
        for i in key_input:
            res.append(activity_rename_mapping[i].lower())
    else:
        res = "none"
        if key_input in activity_rename_mapping.keys():
            return activity_rename_mapping[key_input].lower()
        else:
            return res

    return res


def query_onto_sequence_tao(query):
    if query == "noevent":
        result = "noevent"
        return result
    if "," in query:
        conv = query.split(',')
        query = '+'.join(return_mapping(conv))
    else:
        query = return_mapping(query)
    result = df_tao_onto.loc[df_tao_onto['activity'] == query, "context"].tolist()
    result = ','.join(map(str, result))
    print(query, result)
    return result


def search_context_for_Activity_onto_sparql(activity="ns1:Sitting", step=2):
    # print("Inside ontology Prediction:",activity)
    results = g.query(
        f"""
    Select ?Context where {{ 
        ?Context owl:equivalentClass ?x . 
        ?x ?r ?y . 
        ?y ?a ?b .
        ?b rdf:first/rdf:rest* ?d . 
        ?d ?e {activity} .
    }}

    """
    )
    print("Inside ontology Prediction: ", results, activity)

    for i in results:
        print(i)

    return results


def follow(thefile):
    thefile.seek(0, 2)
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield line


# if __name__ == '__main__':
#     logfile = open("/Users/sudershan/git/mites/mites-io/Mites-Viewer/processing/Mites_Viewer/logs/continuous_events.log", "r")
#     loglines = follow(logfile)
#     for line in loglines:
#         print(line, )

# search_context_for_Activity_onto_sparql()
context = ""
df_real_world_gt_onto_filter = df_real_world_gt_onto[df_real_world_gt_onto['session_id'] == 'p1_' + str(1)]

print (df_real_world_gt_onto_filter.head())

with st.empty():
    for i in range(len(df_real_world_gt_onto_filter)):
        print(df_real_world_gt_onto_filter.iloc[i, 0], df_real_world_gt_onto_filter.iloc[i, 1])
        text = '<p class = "medium-font"> Mites Predictions: </p> ' \
                '<p class = "big-font">' + df_real_world_gt_onto_filter.iloc[i, 2] + '</p> ' \
                '<br> <br>' \
                '<p class = "medium-font"> Context Prediction: </p> ' \
                '<p class = "big-font"> ' + df_real_world_gt_onto_filter.iloc[i, 3] + '</p>'

        st.write(text, unsafe_allow_html=True)
        # st.write(df_real_world_gt_onto.iloc[i, 0],df_real_world_gt_onto.iloc[i, 1], df_real_world_gt_onto.iloc[i, 2],
        #          df_real_world_gt_onto.iloc[i, 3])
        time.sleep(1)
    st.write("✔️ 1 minute over!")
    # for ind in df_real_world_gt_onto:
    #     st.write(df_real_world_gt_onto[ind])
    #     time.sleep(1)


    # for seconds in range(60):
    #     # response = requests.get(url)
    #     # response_dict = json.loads(response.text)
    #     st.write(f"⏳ {seconds} seconds have passed")
    #     st.write(dummy_sample[seconds%len(dummy_sample)])
    #     time.sleep(1)
    # st.write("✔️ 1 minute over!")


# with st.empty():
#     logfile = open(
#         "/Users/sudershan/git/mites/mites-io/Mites-Viewer/processing/Mites_Viewer/logs/continuous_events.log", "r")
#     loglines = follow(logfile)
#     count = 0
#     for line in loglines:
#         # count = 0
#         activity = line.split(',')[2]
#         print(activity, )
#         if count == 50:
#             context = ""
#             count = 0
#             print("current activity", activity)
#             print(activity)
#             if activity in ["Typing", "Writing", "MouseClick"]:
#                 context += "Office_Work"
#             if activity in ["Talking"]:
#                 context += "On_A_Phone_Call, Meeting"
#             if activity == "Coffee Machine":
#                 context += "Meal_Preparation"
#             if context == "":
#                 context = "None"
#             st.write(context)
#         count += 1
#
#     st.write("✔️ 1 minute over!")
