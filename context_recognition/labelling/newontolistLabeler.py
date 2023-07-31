'''
This class that labels contexts based on cluster id, ontological information from a list updated with richer context
'''

import numpy as np
import pandas as pd
import pickle
import os
from scipy.stats import mode
from utils import activity_vec_to_activities


class newontolistLabeler:

    def __init__(self, run_config, logger):
        self.config = run_config
        self.logger = logger
        self.ontolist = newontolist

    def label_clusters(self, context_representations):
        '''
        Label context representations directly to index
        :param context_representations:
        :return:
        '''
        self.context_representations = context_representations

        # Get activities for map to ontology:
        context_activites = []
        activity_labels = np.array(self.config.activity_labels)
        num_activities = len(self.config.activity_labels)

        for context_idx, context_vec in enumerate(self.context_representations):
            activities = []
            for activity_idx, activity_vec in enumerate(np.reshape(context_vec, (-1, num_activities))):
                activity_set = activity_labels[np.where(activity_vec)[0]].tolist()
                activity_set_newontolist = []
                if len(activity_set) > 0:
                    # convert activities to ontological activities
                    for activity in activity_set:
                        activity_newontolist = newontolist.activity_mapping[self.config.dataset][activity]
                        activity_set_newontolist.append(activity_newontolist)
                    activities.append(activity_set_newontolist)
            context_activites.append(activities)

        context_labels = []
        for context_idx, context_activity_map in enumerate(context_activites):
            if len(context_activity_map) > 0:
                onto_label = self.get_onto_label(context_activity_map)
                context_labels.append(f'C{context_idx}__{onto_label}')
            else:
                context_labels.append(f'C{context_idx}__None:1.0')

        # get context labels as cluster labels
        self.cluster_labels = context_labels

        return None

    def get_onto_label(self, context_activity_map):
        """
        Derive context label from ontology based on simple inclusion strategy from parallel and sequential activities
        to use list based ontology, just assume both parallel and sequential relation across
        all activities in context list
        :param context_activity_map: list of activity sets in context representation
        :return:
        """

        # deduplicate activity train in context map
        deduped_activity_map = []
        curr_activity_set = 'None'
        for i in range(len(context_activity_map)):
            if not (curr_activity_set == context_activity_map[i]):
                deduped_activity_map.append(context_activity_map[i])
                curr_activity_set = context_activity_map[i]

        # get context scores based on parallel combinations
        parallel_context_scores = {ctx: 0 for ctx in newontolist.context_mapping.keys()}
        for activity_set in deduped_activity_map:
            for context in newontolist.context_mapping.keys():
                par_context_score = np.sum([activity in newontolist.context_mapping[context] for activity in activity_set])
                parallel_context_scores[context] += par_context_score

        # get context scores based on sequential combinations
        sequential_context_scores = {ctx: 0 for ctx in newontolist.context_mapping.keys()}
        for i in range(len(deduped_activity_map) - 1):
            # get sequential pairs from i->i+1 index
            activity_set_A = deduped_activity_map[i]
            activity_set_B = deduped_activity_map[i + 1]
            for activity_A in activity_set_A:
                for activity_B in activity_set_B:
                    for context in newontolist.context_mapping.keys():
                        is_seq_in_context = (not (activity_A == activity_B)) & (
                                activity_A in newontolist.context_mapping[context]) & (
                                                    activity_B in newontolist.context_mapping[context])
                        sequential_context_scores[context] += is_seq_in_context

        # combine parallel and sequential context scores
        overall_context_scores = {ctx: (sequential_context_scores[ctx] +
                                        parallel_context_scores[ctx])
                                  for ctx in newontolist.context_mapping.keys()}
        onto_label = ''
        sum_context_scores = np.sum(list(overall_context_scores.values()))
        for ctx in overall_context_scores.keys():
            if overall_context_scores[ctx] > 0:
                onto_label += f'{ctx}:{overall_context_scores[ctx] / sum_context_scores:.1f}_'
        if onto_label == '':
            onto_label = 'None:1.0_'
        onto_label = onto_label[:-1]  # remove last underscore
        return onto_label

    def get_cluster_label(self, cluster_id: int):
        '''
        Get cluster label based on ids
        :param cluster_id: id of cluster to find label for
        :return: cluster label string
        '''
        return self.cluster_labels[cluster_id]

    def save(self):
        exp_dir = f'{self.config.cache_dir}/{self.config.experiment}'
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        context_label_object = {
            'context_representations': self.context_representations,
            'cluster_labels': self.cluster_labels
        }
        labeler_file = f'{self.config.cache_dir}/{self.config.experiment}/newontolist_labels_{self.config.model_re}' \
                       f'_{self.config.model_cluster}.pb'
        pickle.dump(context_label_object, open(labeler_file, 'wb'))
        return None

    def load(self):
        labeler_file = f'{self.config.cache_dir}/{self.config.experiment}/newontolist_labels_{self.config.model_re}' \
                       f'_{self.config.model_cluster}.pb'
        if os.path.exists(labeler_file):
            context_label_object = pickle.load(open(labeler_file, 'rb'))
            self.cluster_labels = context_label_object['cluster_labels']
            self.context_representations = context_label_object['context_representations']
            return True
        else:
            return False


class newontolist:
    context_mapping = {

        'None': ['none'],
        'Amusement': ['lying', 'sitting', 'standing_mall', 'watching_tv', 'surfing_internet', 'singing', 'talking',
                      'meeting_friends'],
        'ComingIn': ['walking'],
        'Commuting': ['sitting', 'walking', 'cycling', 'driving', 'in_car', 'climbingstairs', 'descendingstairs'],
        'Exercising': ['sitting', 'walking', 'running', 'cycling', 'exercising', 'jogging', 'climbingstairs',
                       'descendingstairs', 'standing'],
        'GoingOut': ['walking', 'cycling'],
        'HavingMeal': ['sitting', 'drinking', 'eating'],
        'HouseWork': ['sitting', 'walking', 'cleaning_home', 'laundry_home', 'dishes_home', 'standing'],
        'Inactivity': ['lying', 'sitting'],
        'Meal_Preparation': [],
        'OfficeWork': ['sitting', 'lab_work', 'class_work', 'office_work'],
        'InAMeeting': ['sitting', 'meeting_work', 'talking', 'meeting_coworkers', 'meeting_friends'],
        # 'OnAPhoneCall':['sitting'],
        'PhoneCall': ['sitting', 'walking'],
        'PreparingMeal': ['sitting', 'cooking', 'standing'],
        'Relaxing': ['lying', 'drinking', 'watching_tv'],
        'Sleeping': ['lying', 'sleeping'],
        'UsingBathroom': ['sitting', 'walking', 'shower', 'toilet', 'grooming', 'dressing_up', 'standing'],
    }

    activity_mapping = {
        'extrasensory': {
            "Lying down": 'lying',
            "Sitting": 'sitting',
            "Walking": 'walking',
            "Running": 'running',
            "Bicycling": 'cycling',
            "Sleeping": 'sleeping',
            "Lab work": 'lab_work',
            "In class": 'class_work',
            "In a meeting": 'meeting_work',
            "Drive - I'm the driver": 'driving',
            "Drive - I'm a passenger": 'in_car',
            "Exercise": 'exercising',
            "Cooking": 'cooking',
            "Shopping": 'standing_mall',
            "Strolling": 'jogging',
            "Drinking (alcohol)": 'drinking',
            "Bathing - shower": 'shower',
            "Cleaning": 'cleaning_home',
            "Doing laundry": 'laundry_home',
            "Washing dishes": 'dishes_home',
            "WatchingTV": 'watching_tv',
            "Surfing the internet": 'surfing_internet',
            "Singing": 'singing',
            "Talking": 'talking',
            "Computer work": 'office_work',
            "Eating": 'eating',
            "Toilet": 'toilet',
            "Grooming": 'grooming',
            "Dressing": 'dressing_up',
            "Stairs - going up": 'climbingstairs',
            "Stairs - going down": 'descendingstairs',
            "Standing": 'standing',
            "With co-workers": 'meeting_coworkers',
            "With friends": 'meeting_friends',

        },
        # 'aruba': {
        #     "None": 'none',
        #     "Sleeping": 'lying_home',
        #     "Bed_to_Toilet": 'walking_home',
        #     "Meal_Preparation": 'standing_home',
        #     "Relax": 'sitting_home',
        #     "Housekeeping": 'sweeping_home',
        #     "Eating": 'eating',
        #     "Work": 'sitting_office',
        #     "Wash_Dishes": 'sweeping_home',
        #     "Leave_Home": 'sitting_transport',
        #     "Resperate": 'sitting_home',
        #     "Enter_Home": 'walking_home'
        # },
        # 'aruba_w_location': {
        #     "None_Master Bathroom": "standing_home",
        #     "Sleeping_Master Bathroom": "lying",
        #     "Bed_to_Toilet_Master Bathroom": "walking_home",
        #     "Meal_Preparation_Master Bathroom": "eating",
        #     "Relax_Master Bathroom": "sitting_home",
        #     "Housekeeping_Master Bathroom": "sweeping_home",
        #     "Eating_Master Bathroom": "eating",
        #     "Work_Master Bathroom": "sweeping_home",
        #     "None_Living Room": "sitting_home",
        #     "Sleeping_Living Room": "lying",
        #     "Bed_to_Toilet_Living Room": "standing_home",
        #     "Meal_Preparation_Living Room": "eating",
        #     "Relax_Living Room": "sitting_home",
        #     "Housekeeping_Living Room": "sweeping_home",
        #     "Eating_Living Room": "eating",
        #     "None_Kitchen": "eating",
        #     "Sleeping_Kitchen": "standing_home",
        #     "Meal_Preparation_Kitchen": "eating",
        #     "Relax_Kitchen": "sitting_home",
        #     "Housekeeping_Kitchen": "sweeping_home",
        #     "Eating_Kitchen": "eating",
        #     "Wash_Dishes_Kitchen": "sweeping_home",
        #     "Leave_Home_Kitchen": "standing_home",
        #     "Resperate_Kitchen": "sitting_home",
        #     "None_Junction": "none",
        #     "Sleeping_Junction": "sitting_mall",
        #     "Meal_Preparation_Junction": "eating",
        #     "Relax_Junction": "lying",
        #     "Housekeeping_Junction": "sweeping_yard",
        #     "Eating_Junction": "eating",
        #     "Wash_Dishes_Junction": "sweeping_yard",
        #     "Leave_Home_Junction": "walking_yard",
        #     "Resperate_Junction": "standing_yard",
        #     "None_Second Bedroom": "none",
        #     "Sleeping_Second Bedroom": "lying_home",
        #     "Meal_Preparation_Second Bedroom": "eating",
        #     "Relax_Second Bedroom": "sitting_home",
        #     "Housekeeping_Second Bedroom": "sweeping_home",
        #     "Eating_Second Bedroom": "eating",
        #     "Wash_Dishes_Second Bedroom": "sweeping_home",
        #     "Work_Second Bedroom": "sitting_home",
        #     "Resperate_Second Bedroom": "standing_home",
        #     "None_Corridor": "none",
        #     "Sleeping_Corridor": "lying",
        #     "Meal_Preparation_Corridor": "eating",
        #     "Relax_Corridor": "lying",
        #     "Housekeeping_Corridor": "sweeping_home",
        #     "Eating_Corridor": "eating",
        #     "Wash_Dishes_Corridor": "sweeping_home",
        #     "Work_Corridor": "sweeping_home",
        #     "Leave_Home_Corridor": "none",
        #     "Enter_Home_Corridor": "none",
        #     "None_Outside": "none",
        #     "Sleeping_Outside": "lying",
        #     "Meal_Preparation_Outside": "eating",
        #     "Relax_Outside": "sitting_mall",
        #     "Wash_Dishes_Outside": "walking",
        #     "Work_Outside": "sitting_office",
        #     "Leave_Home_Outside": "walking_home",
        #     "Enter_Home_Outside": "walking_home",
        #     "None_Office": "none",
        #     "Sleeping_Office": "sitting_ofice",
        #     "Meal_Preparation_Office": "eating_office",
        #     "Relax_Office": "lying",
        #     "Housekeeping_Office": "sweeping_office",
        #     "Wash_Dishes_Office": "sweeping_office",
        #     "Work_Office": "sitting_office",
        #     "Enter_Home_Office": "nonw",
        #     "Resperate_Office": "sitting_office",
        #     "None_Second Bathroom": "none",
        #     "Sleeping_Second Bathroom": "lying",
        #     "Meal_Preparation_Second Bathroom": "none",
        #     "Relax_Second Bathroom": "none",
        #     "Housekeeping_Second Bathroom": "sweeping_home",
        #     "Eating_Second Bathroom": "eating",
        #     "Wash_Dishes_Second Bathroom": "sweeping_home",
        #     "Work_Second Bathroom": "sitting",
        #     "Leave_Home_Second Bathroom": "walking_home",
        #     "Enter_Home_Second Bathroom": "walking_home"
        # },
        # 'witham': {
        #     "Outside": 'walking',
        #     "Eating": 'eating',
        #     "Other": 'none',
        #     "Reading": 'sitting_mall',
        #     "Talking": 'sitting',
        #     "Writing": 'sitting_office',
        #     "Video": 'sitting_office',
        #     "Phonecall": 'walking_home',
        #     "Sleeping": 'lying',
        #     "Toilett": 'sitting_home'
        # },
        # 'opportunity': {
        #     "Walk": "walking",
        #     "Stand": "standing",
        #     "Sit": "sitting",
        #     "Lie": "lying",
        #     "reach": "standing",
        #     "open": "standing_home",
        #     "close": "standing_home",
        #     "release": "standing_home",
        #     "move": "walking_home",
        #     "bite": "walking_home",
        #     "Lazychair": "sitting_home",
        #     "Fridge": "eating",
        #     "Drawer2 (middle)": "eating",
        #     "Drawer3 (lower)": "eating",
        #     "Cup": "eating",
        #     "Glass": "eating",
        #     "Sugar": "eating",
        #     "Drawer1 (top)": "walking_home",
        #     "Door1": "walking_home",
        #     "Plate": "eating",
        #     "Cheese": "eating",
        #     "Bread": "eating",
        #     "Knife salami": "eating_home",
        #     "Salami": "eating",
        #     "Bottle": "eating",
        #     "Dishwasher": "standing_home",
        #     "stir": "eating",
        #     "sip": "eating",
        #     "spread": "eating",
        #     "cut": "walking_home",
        #     "Door2": "standing_home",
        #     "Spoon": "eating",
        #     "Milk": "eating",
        #     "Knife cheese": "eating",
        #     "Open Dishwasher": "standing_home",
        #     "Close Dishwasher": "standing_home",
        #     "Open Fridge": "standing_home",
        #     "Close Fridge": "standing_home",
        #     "Open Door 2": "standing_home",
        #     "Open Door 1": "standing_home",
        #     "Open Drawer 1": "standing",
        #     "Close Drawer 1": "standing",
        #     "Open Drawer 2": "standing",
        #     "Close Drawer 2": "standing",
        #     "Open Drawer 3": "standing",
        #     "Close Drawer 3": "standing",
        #     "Close Door 1": "standing",
        #     "Close Door 2": "standing",
        #     "Drink from Cup": "eating",
        #     "lock": "walking_home",
        #     "clean": "sweeping_home",
        #     "unlock": "sweeping_home",
        #     "Chair": "sitting_home",
        #     "Table": "sitting_home",
        #     "Switch": "standing_home",
        #     "Toggle Switch": "standing_home",
        #     "Clean Table": "sweeping_home"
        # },
        # 'casas': {
        #     "Step_Out": 'standing_transport',
        #     "Other_Activity": 'none',
        #     "Toilet": 'walking_home',
        #     "Phone": 'sitting_home',
        #     "Personal_Hygiene": 'standing_home',
        #     "Leave_Home": 'sitting_transport',
        #     "Enter_Home": 'walking_home',
        #     "Relax": 'lying',
        #     "Sleep_Out_Of_Bed": 'lying_home',
        #     "Drink": 'eating',
        #     "Watch_TV": 'sitting_home',
        #     "Dress": 'standing_home',
        #     "Evening_Meds": 'sitting_home',
        #     "Wake_Up": 'lying_home',
        #     "Read": 'sitting_office',
        #     "Morning_Meds": 'sitting_home',
        #     "Cook_Breakfast": 'standing_home',
        #     "Eat_Breakfast": 'eating_home',
        #     "Bathe": 'standing_home',
        #     "Cook_Lunch": 'standing_home',
        #     "Eat_Lunch": 'eating',
        #     "Wash_Lunch_Dishes": 'standing_home',
        #     "Go_To_Sleep": 'lying_home',
        #     "Sleep": 'lying_home',
        #     "Bed_Toilet_Transition": 'walking_home',
        #     "Wash_Breakfast_Dishes": 'standing_home',
        #     "Work_At_Table": 'sitting_office',
        #     "Groom": 'standing_home',
        #     "Cook": 'standing_home',
        #     "Eat": 'eating',
        #     "Cook_Dinner": 'standing_home',
        #     "Eat_Dinner": 'eating_home',
        #     "Wash_Dinner_Dishes": 'sweeping_home',
        #     "Wash_Dishes": 'sweeping_home',
        #     "Entertain_Guests": 'sitting_home',
        #     "Take_Medicine": 'sitting_home',
        #     "Work": 'sitting_office',
        #     "Exercise": 'stretching',
        #     "Work_On_Computer": 'sitting_office',
        #     "Nap": 'lying_home',
        #     "Work_At_Desk": 'sitting_office',
        #     "Laundry": 'sweeping_home'
        # }
    }
