'''
This class directly labels contexts based on cluster id, without any ontological information
'''

import numpy as np
import pandas as pd
import pickle
import os


class directLabeler:

    def __init__(self, run_config, logger):
        self.config = run_config
        self.logger = logger
        self.ontolist = directontolist

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


        # Get activities for map to ontology:
        context_activites = []
        activity_labels = np.array(self.config.activity_labels)
        num_activities = len(self.config.activity_labels)

        for context_idx, context_vec in enumerate(self.context_representations):
            activities = []
            for activity_idx, activity_vec in enumerate(np.reshape(context_vec, (-1, num_activities))):
                activity_set = activity_labels[np.where(activity_vec)[0]].tolist()
                activity_set_directontolist = []
                if len(activity_set) > 0:
                    # convert activities to ontological activities
                    for activity in activity_set:
                        activity_directontolist = directontolist.activity_mapping[self.config.dataset][activity]
                        if not (activity_directontolist == 'none'):
                            activity_set_directontolist.append(activity_directontolist)
                    if len(activity_set_directontolist) > 0:
                        activities.append(activity_set_directontolist)
            context_activites.append(activities)

        context_labels = []
        self.logger.info("Cluster labelling based on activity train")
        for context_idx, context_activity_map in enumerate(context_activites):
            if len(context_activity_map) > 0:
                deduped_activity_map = []
                curr_activity_set = 'None'
                for i in range(len(context_activity_map)):
                    if not (curr_activity_set == context_activity_map[i]):
                        deduped_activity_map.append('+'.join(sorted(context_activity_map[i])))
                        curr_activity_set = context_activity_map[i]
                activity_train = '>'.join(deduped_activity_map)
                context_labels.append(f'C{context_idx}({activity_train})__None:1.0')
            else:
                context_labels.append(f'C{context_idx}()__None:1.0')

        self.cluster_labels = context_labels
        self.logger.info(f"Final Cluster Labels:\n {context_labels}")

        # get context labels as cluster labels
        self.cluster_labels = context_labels
        self.logger.info(f"Final Cluster Labels:\n {context_labels}")
        return None

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
        labeler_file = f'{self.config.cache_dir}/{self.config.experiment}/direct_labels_{self.config.model_re}' \
                       f'_{self.config.model_cluster}_{self.config.cnet_n_clusters}.pb'
        pickle.dump(context_label_object, open(labeler_file, 'wb'))
        return None

    def load(self):
        labeler_file = f'{self.config.cache_dir}/{self.config.experiment}/direct_labels_{self.config.model_re}' \
                       f'_{self.config.model_cluster}_{self.config.cnet_n_clusters}.pb'
        if os.path.exists(labeler_file):
            context_label_object = pickle.load(open(labeler_file, 'rb'))
            self.cluster_labels = context_label_object['cluster_labels']
            self.context_representations = context_label_object['context_representations']
            return True
        else:
            return False


class directontolist:
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
        'casas': {
            "Step_Out": 'step_out',
            "Other_Activity": 'none',
            "Toilet": 'toilet',
            "Phone": 'onphone',
            "Personal_Hygiene": 'grooming',
            "Leave_Home": 'step_out',
            "Enter_Home": 'step_in',
            "Relax": 'lying',
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
            "boil_water":"boil_water",
            "clean_with_water":"clean_with_water",
            "cut":"cut",
            "cut_bread":"cut_bread",
            "drinkfrom_bottle":"drinkfrom_bottle",
            "drinkfrom_can":"drinkfrom_can",
            "drinkfrom_cup":"drinkfrom_cup",
            "drinkfrom_glass":"drinkfrom_glass",
            "dry_up":"dry_up",
            "dump_in_trash":"dump_in_trash",
            "eat_at_table":"eat_at_table",
            "eat_snack":"eat_snack",
            "enter":"enter",
            "get_up":"get_up",
            "get_water":"get_water",
            "insert_tea_bag":"insert_tea_bag",
            "lay_down":"lay_down",
            "leave":"leave",
            "none":"none",
            "pour_grains":"pour_grains",
            "pour_water":"pour_water",
            "pourfrom_bottle":"pourfrom_bottle",
            "pourfrom_can":"pourfrom_can",
            "pourfrom_kettle":"pourfrom_kettle",
            "put_something_in_sink":"put_something_in_sink",
            "put_something_on_table":"put_something_on_table",
            "read":"read",
            "sit_down":"sit_down",
            "spread_jam_or_butter":"spread_jam_or_butter",
            "stir":"stir",
            "stir_coffee/tea":"stir_coffee/tea",
            "take_ham":"take_ham",
            "take_pills":"take_pills",
            "take_something_off_table":"take_something_off_table",
            "use_cupboard":"use_cupboard",
            "use_drawer":"use_drawer",
            "use_fridge":"use_fridge",
            "use_glasses":"use_glasses",
            "use_laptop":"use_laptop",
            "use_oven":"use_oven",
            "use_stove":"use_stove",
            "use_tablet":"use_tablet",
            "use_telephone":"use_telephone",
            "walk":"walk",
            "watch_tv":"watch_tv",
            "wipe_table":"wipe_table",
            "write":"write",
        }
    }
