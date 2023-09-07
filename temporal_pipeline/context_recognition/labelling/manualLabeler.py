'''
This class that labels contexts manually with a human observer
'''

import numpy as np
import pandas as pd
import pickle
import os
import json
from scipy.stats import mode
from utils import activity_vec_to_activities


class manualontolistLabeler:

    def __init__(self, run_config, logger):
        self.config = run_config
        self.logger = logger
        self.ontolist = manualontolist

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
                activity_set_manualontolist = []
                if len(activity_set) > 0:
                    # convert activities to ontological activities
                    for activity in activity_set:
                        activity_manualontolist = manualontolist.activity_mapping[self.config.dataset][activity]
                        if not (activity_manualontolist == 'none'):
                            activity_set_manualontolist.append(activity_manualontolist)
                    if len(activity_set_manualontolist) > 0:
                        activities.append(activity_set_manualontolist)
            context_activites.append(activities)

        # Load previous manual labels
        if os.path.exists(self.config.manual_labels_backup):
            labels_history = pickle.load(open(self.config.manual_labels_backup, 'rb'))
        else:
            labels_history = {}

        context_labels = []
        self.logger.info("Cluster Manual Labelling")
        context_list = sorted(list(manualontolist.context_mapping.keys()))
        self.logger.info("List of contexts with numbering")
        context_list_str = ';'.join([f"({i}){ctx}" for i, ctx in enumerate(context_list)])
        print(context_list_str)
        for context_idx, context_activity_map in enumerate(context_activites):
            if len(context_activity_map) > 0:
                deduped_activity_map = []
                curr_activity_set = 'None'
                for i in range(len(context_activity_map)):
                    if not (curr_activity_set == context_activity_map[i]):
                        deduped_activity_map.append(','.join(sorted(context_activity_map[i])))
                        curr_activity_set = context_activity_map[i]

                activity_map_set = set(deduped_activity_map)  # todo: sequentiality is not taken in consideration
                backup_label = 'None:1.0'
                for ctx_key in labels_history.keys():
                    if activity_map_set in labels_history[ctx_key]:
                        backup_label = ctx_key

                print('->'.join(deduped_activity_map))
                print(f'Backup Label: {backup_label}')
                ctx_input = input(
                    "Press Enter to accept Backup Label, or provide comma separated context ids for this cluster:")
                if ctx_input == '':
                    manual_label = backup_label
                else:
                    contexts = [context_list[int(idx_ctx)] for idx_ctx in ctx_input.split(",")]
                    context_prob = 1 / len(contexts)
                    manual_label = '_'.join([f"{ctx}:{context_prob:.1f}" for ctx in contexts])
                    if manual_label not in labels_history.keys():
                        labels_history[manual_label] = []
                    labels_history[manual_label].append(activity_map_set)
                context_labels.append(f'C{context_idx}__{manual_label}')
            else:
                context_labels.append(f'C{context_idx}__None:1.0')

        # get context labels as cluster labels
        self.cluster_labels = context_labels
        self.logger.info(f"Final Cluster Labels:\n {context_labels}")

        # dump new manual labels backup file
        pickle.dump(labels_history, open(self.config.manual_labels_backup, 'wb'))

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
        parallel_context_scores = {ctx: 0 for ctx in manualontolist.context_mapping.keys()}
        for activity_set in deduped_activity_map:
            for context in manualontolist.context_mapping.keys():
                par_context_score = np.sum(
                    [activity in manualontolist.context_mapping[context] for activity in activity_set])
                parallel_context_scores[context] += par_context_score

        # get context scores based on sequential combinations
        sequential_context_scores = {ctx: 0 for ctx in manualontolist.context_mapping.keys()}
        for i in range(len(deduped_activity_map) - 1):
            # get sequential pairs from i->i+1 index
            activity_set_A = deduped_activity_map[i]
            activity_set_B = deduped_activity_map[i + 1]
            for activity_A in activity_set_A:
                for activity_B in activity_set_B:
                    for context in manualontolist.context_mapping.keys():
                        is_seq_in_context = (not (activity_A == activity_B)) & (
                                activity_A in manualontolist.context_mapping[context]) & (
                                                    activity_B in manualontolist.context_mapping[context])
                        sequential_context_scores[context] += is_seq_in_context

        # combine parallel and sequential context scores
        overall_context_scores = {ctx: (sequential_context_scores[ctx] +
                                        parallel_context_scores[ctx])
                                  for ctx in manualontolist.context_mapping.keys()}
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
        labeler_file = f'{self.config.cache_dir}/{self.config.experiment}/manualontolist_labels_{self.config.model_re}' \
                       f'_{self.config.model_cluster}.pb'
        pickle.dump(context_label_object, open(labeler_file, 'wb'))
        return None

    def load(self):
        labeler_file = f'{self.config.cache_dir}/{self.config.experiment}/manualontolist_labels_{self.config.model_re}' \
                       f'_{self.config.model_cluster}.pb'
        if os.path.exists(labeler_file):
            context_label_object = pickle.load(open(labeler_file, 'rb'))
            self.cluster_labels = context_label_object['cluster_labels']
            self.context_representations = context_label_object['context_representations']
            return True
        else:
            return False


class manualontolist:
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
        'HouseWork': ['sitting', 'walking', 'cleaning_home', 'laundry_home', 'dishes_home', 'standing', 'taking_meds'],
        'Inactivity': ['lying', 'sitting'],
        'Meal_Preparation': [],
        'OfficeWork': ['sitting', 'lab_work', 'class_work', 'office_work'],
        'InAMeeting': ['sitting', 'meeting_work', 'talking', 'meeting_coworkers', 'meeting_friends'],
        'OnAPhoneCall': ['sitting', 'walking', 'onphone'],
        'PhoneCall': ['sitting', 'walking', 'onphone'],
        'PreparingMeal': ['sitting', 'cooking', 'standing'],
        'Relaxing': ['lying', 'drinking', 'watching_tv'],
        'Sleeping': ['lying', 'sleeping'],
        'UsingBathroom': ['sitting', 'walking', 'shower', 'toilet', 'grooming', 'dressing_up', 'standing'],

        # TSU Specific Contexts(not activites because it is manual labelling)
        'tsuBreakfast': [],
        'tsuCook': [],
        'tsuMakecoffee': [],
        'tsuMaketea': [],
        'tsuRelax': [],
        'tsuCleandishes': [],
    }

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
        },
        'realworld': {
            "coffeemachine": "coffeemachine",
            "doorknock": "doorknock",
            "dooropen": "dooropen",
            "eating": "eating",
            "houseWork": "houseWork",
            "jumping": "jumping",
            "mouse": "mouse",
            "mouseclick": "mouseclick",
            "phonering": "phonering",
            "running": "running",
            "sweeping": "sweeping",
            "talking": "talking",
            "tv": "tv",
            "typing": "typing",
            "vacuum": "vacuum",
            "writing": "writing",
        }
    }