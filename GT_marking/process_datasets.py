import pandas as pd
import numpy as np

activity_mapping = {
        'extrasensory': {
            "Lying down": 'lying',
            "Sitting": 'sitting',
            "Walking": 'walking',
            "Running": 'running',
            "Bicycling": 'cycling',
            "Sleeping": 'sleeping',
            "Lab work": 'office_work',
            "In class": 'office_work',
            "In a meeting": 'meeting',
            "Drive - I'm the driver": 'driving',
            "Drive - I'm a passenger": 'driving',
            "Exercise": 'exercising',
            "Cooking": 'cooking',
            "Shopping": 'shopping',
            "Strolling": 'running',
            "Drinking (alcohol)": 'drinking',
            "Bathing - shower": 'shower',
            "Cleaning": 'cleaning',
            "Doing laundry": 'laundry',
            "Washing dishes": 'clean_dishes',
            "WatchingTV": 'watching_tv',
            "Surfing the internet": 'surfing_internet',
            "Singing": 'singing',
            "Talking": 'talking',
            "Computer work": 'office_work',
            "Eating": 'eating',
            "Toilet": 'toilet',
            "Grooming": 'grooming',
            "Dressing": 'dressing_up',
            "Stairs - going up": 'stairs',
            "Stairs - going down": 'stairs',
            "Standing": 'standing',
            "With co-workers": 'meeting_coworkers',
            "With friends": 'meeting_friends',

        },
        'casas': {
            "Step_Out": 'step_out',
            "Other_Activity": 'none',
            "Toilet": 'toilet',
            "Phone": 'onphone',
            "Personal_Hygiene": 'grooming',
            "Leave_Home": 'step_out',
            "Enter_Home": 'step_in',
            "Relax": 'watching_tv',
            "Sleep_Out_Of_Bed": 'lying',
            "Drink": 'drinking',
            "Watch_TV": 'watching_tv',
            "Dress": 'dressing_up',
            "Evening_Meds": 'taking_meds',
            "Wake_Up": 'wakingup',
            "Read": 'reading',
            "Morning_Meds": 'taking_meds',
            "Cook_Breakfast": 'cooking',
            "Eat_Breakfast": 'eating',
            "Bathe": 'shower',
            "Cook_Lunch": 'cooking',
            "Eat_Lunch": 'eating',
            "Wash_Lunch_Dishes": 'dishes_home',
            "Go_To_Sleep": 'sleeping',
            "Sleep": 'sleeping',
            "Bed_Toilet_Transition": 'toilet',
            "Wash_Breakfast_Dishes": 'dishes_home',
            "Work_At_Table": 'office_work',
            "Groom": 'grooming',
            "Cook": 'cooking',
            "Eat": 'eating',
            "Cook_Dinner": 'cooking',
            "Eat_Dinner": 'eating',
            "Wash_Dinner_Dishes": 'dishes_home',
            "Wash_Dishes": 'dishes_home',
            "Entertain_Guests": 'meeting_friends',
            "Take_Medicine": 'taking_meds',
            "Work": 'office_work',
            "Exercise": 'exercising',
            "Work_On_Computer": 'office_work',
            "Nap": 'lying',
            "Work_At_Desk": 'office_work',
            "Laundry": 'laundry_home'
        },
        'tsu': {
            "boil_water": "boil_water",
            "clean_with_water": "clean_with_water",
            "cut": "cut_cook",
            "cut_bread": "cut_bread",
            "drinkfrom_bottle": "drink_cold",
            "drinkfrom_can": "drink_cold",
            "drinkfrom_cup": "drink_hot",
            "drinkfrom_glass":"drink_cold",
            "dry_up": "dry_up",
            "dump_in_trash": "dump_in_trash",
            "eat_at_table": "eat_food",
            "eat_snack": "eat_snack",
            "enter": "enter",
            "get_up": "get_up",
            "get_water": "get_water",
            "insert_tea_bag": "insert_tea_bag",
            "lay_down": "lay_down",
            "leave": "leave",
            "none": "none",
            "pour_grains": "pour_grains",
            "pour_water": "pour_water",
            "pourfrom_bottle": "pour_cold",
            "pourfrom_can": "pour_cold",
            "pourfrom_kettle": "pour_hot",
            "put_something_in_sink": "put_in_sink",
            "put_something_on_table": "put_on_table",
            "read": "read",
            "sit_down": "sit_down",
            "spread_jam_or_butter": "spread_jam_or_butter",
            "stir": "stir_cook",
            "stir_coffee/tea": "stir_drink",
            "take_ham": "take_ham",
            "take_pills": "take_meds",
            "take_something_off_table": "take_off_table",
            "use_cupboard": "use_furniture",
            "use_drawer": "use_furniture",
            "use_fridge": "use_furniture",
            "use_glasses": "use_glasses",
            "use_laptop": "use_pc",
            "use_oven": "use_kitchen_utility",
            "use_stove": "use_kitchen_utility",
            "use_tablet": "use_pc",
            "use_telephone": "use_telephone",
            "walk": "walk",
            "watch_tv": "watch_tv",
            "wipe_table": "clean_table",
            "write": "write",
        }
    }

def process_extrasensory(datasetfile):
    '''
    Returns a dataset in format of id, timestamp, datetime, activity
    :param datasetfile:
    :return:
    '''
    activity_list = ["Lying down","Sitting", "Walking", "Running", "Bicycling", "Sleeping", "Lab work", "In class",
              "In a meeting", "Drive - I'm the driver", "Drive - I'm a passenger", "Exercise", "Cooking",
              "Shopping", "Strolling", "Drinking (alcohol)","Bathing - shower", "Cleaning", "Doing laundry",
              "Washing dishes", "WatchingTV", "Surfing the internet", "Singing", "Talking", "Computer work",
              "Eating", "Toilet", "Grooming", "Dressing", "Stairs - going up", "Stairs - going down", "Standing",
              "With co-workers", "With friends"]
    df_data = pd.read_csv(datasetfile)
    df_data = df_data[['uuid','timestamp']+activity_list]
    df_data = df_data.melt(id_vars=['uuid','timestamp'],ignore_index=True)
    df_data['datetime'] = pd.to_datetime(df_data['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert('America/Los_Angeles')
    df_data = df_data[df_data.value==1]
    df_data.columns= ['id','timestamp','activity','value','datetime']
    df_data =df_data[['id','timestamp','datetime','activity']].sort_values(by=['id','timestamp'])
    return df_data


def process_opportunity(datasetfile):
    '''
    Returns a dataset in format of id, timestamp, datetime, activity
    :param datasetfile:
    :return:
    '''
    df_data = pd.read_csv(datasetfile)
    return df_data


def process_casas(datasetfile):
    '''
    Returns a dataset in format of id, timestamp, datetime, activity
    :param datasetfile:
    :return:
    '''

    df_data = pd.read_csv(datasetfile)
    df_data['activity_name'] = df_data.Activity.apply(
        lambda x: x.strip().replace('r1.', '').replace('r2.', ''))
    df_data['activity_name'] = df_data['activity_name'].apply(lambda x: activity_mapping['casas'][x])
    df_data = df_data[~(df_data.activity_name=='none')]
    df_data['timestamp'] = pd.to_datetime(df_data['DateTime'], format='%Y-%m-%d %H:%M:%S.%f')
    df_data['timestamp'] = pd.to_numeric(df_data['timestamp'].values) / 10 ** 9
    df_data['datetime'] = pd.to_datetime(df_data['timestamp'], unit='s')
    df_data = df_data[['Home','timestamp','datetime','activity_name']]
    df_data.columns = ['id','timestamp','datetime','activity']
    df_data = df_data.sort_values(by=['id','timestamp'])
    return df_data


def process_aruba(datasetfile):
    '''
    Returns a dataset in format of id, timestamp, datetime, activity
    :param datasetfile:
    :return:
    '''
    df_data = pd.read_csv(datasetfile)
    df_data['activity_name'] = df_data.apply(
        lambda row: f"{row['activity_name'].strip()}", axis=1)
    df_data = df_data[~(df_data.activity_name=='None')]
    df_data['timestamp'] = pd.to_datetime(df_data['timestamp'], format='%Y-%m-%d %H:%M:%S')
    df_data['timestamp'] = pd.to_numeric(df_data['timestamp'].values) / 10 ** 9
    df_data['datetime'] = pd.to_datetime(df_data['timestamp'], unit='s')
    df_data['id'] = 'None'
    df_data = df_data[['id','timestamp','datetime','activity_name']]
    df_data.columns = ['id','timestamp','datetime','activity']
    df_data = df_data.sort_values(by=['id','timestamp'])

    return df_data


def process_witham(datasetfile):
    '''
    Returns a dataset in format of id, timestamp, datetime, activity
    :param datasetfile:
    :return:
    '''
    df_data = pd.read_csv(datasetfile)
    df_data['activity_name'] = df_data.apply(
        lambda row: f"{row['activity_name'].strip()}", axis=1)
    df_data['timestamp'] = pd.to_datetime(df_data['timestamp'], format='%Y-%m-%d %H:%M:%S')
    df_data['timestamp'] = pd.to_numeric(df_data['timestamp'].values) / 10 ** 9
    df_data['datetime'] = pd.to_datetime(df_data['timestamp'], unit='s')
    df_data['id'] = 'None'
    df_data = df_data[['id','timestamp','datetime','activity_name']]
    df_data.columns = ['id','timestamp','datetime','activity']
    df_data = df_data.sort_values(by=['id','timestamp'])
    return df_data
