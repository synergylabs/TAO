import rdflib
import pandas as pd
from itertools import combinations

g = rdflib.Graph()

g.parse("context-v2.n3", format="n3")
g.bind("ns1", "http://www.miningminds.re.kr/lifelog/context/context-v1.owl#")
# g.bind("ns1", "../ontological_models/context-v2.owl")
# g.serialize("../ontological_models/context-v2.ttl")

activity_rename_mapping = {'extrasensory': {
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
    "Shopping": 'Standing_Mall',
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

},
'casas': {
    "Step_Out": 'StepOut',
    "Other_Activity":'None',
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
    "Laundry": 'VacuumHome'
}}


contexts_list = {
    'Amusement': ['context_dancing_mall_happiness', 'context_sitting_mall_happiness',
                  'context_walking_mall_happiness'],
    'Commuting': ['context_sitting_transport', 'context_sitting_transport_fear', 'context_standing_transport'],
    'Exercising': ['context_climbingstairs_gym', 'context_climbingstairs_gym_happiness', 'context_climbingstairs_home',
                   'context_climbingstairs_home_neutral', 'context_climbingstairs_office', 'context_cycling_gym',
                   'context_cycling_gym_neutral', 'context_cycling_outdoors', 'context_descendingstairs_gym_happiness',
                   'context_descendingstairs_home', 'context_descendingstairs_office_neutral',
                   'context_hiking_outdoors', 'context_hiking_outdoors_happiness', 'context_jogging_gym',
                   'context_jogging_gym_happiness', 'context_jogging_outdoors', 'context_jogging_outdoors_neutral',
                   'context_running_gym', 'context_running_outdoors', 'context_stretching_gym',
                   'context_stretching_home', 'context_stretching_home_neutral', 'context_stretching_office',
                   'context_stretching_outdoors', 'context_stretching_outdoors_happiness'],
    'HavingMeal': ['context_eating_home', 'context_eating_home_disgust', 'context_eating_restaurant',
                   'context_eating_restaurant_neutral', 'context_eating_restaurant_surprise',
                   'context_sitting_restaurant', 'context_sitting_restaurant_happiness'],
    'HouseWork': ['context_standing_home', 'context_standing_home_anger', 'context_standing_home_boredom',
                  'context_standing_home_disgust', 'context_standing_home_happiness', 'context_standing_home_neutral',
                  'context_sweeping_home', 'context_walking_home', 'context_walking_home_boredom'],
    'Inactivity': ['context_lying_home_happiness', 'context_lying_office', 'context_lying_office_neutral',
                   'context_sitting_home', 'context_sitting_mall', 'context_sitting_mall_boredom',
                   'context_sitting_mall_sadness', 'context_sitting_office_sadness', 'context_sitting_restaurant_anger',
                   'context_standing_home_surprise'],
    'Relaxing': ['context_sitting_office', 'context_sitting_office_anger', 'context_sitting_office_boredom',
                   'context_sitting_office_disgust', 'context_sitting_office_happiness',
                   'context_sitting_office_neutral'],
    'Sleeping': ['context_lying_home', 'context_lying_home_neutral'],
}


def get_contexts(graph, subject):
    contexts = list()
    # sub = "ns1:"+subject+
    # print (sub)
    # res = graph.query("""
    #     SELECT
    #     DISTINCT ?context
    #     WHERE {
    #         %s (<sn>|!<sn>)+ ?context .
    #         ?context rdfs:subClassOf ns1:Context .
    #     }
    # """ % subject)
    #
    # for row in res:
    #     print(row)
    #     contexts.append(graph.namespace_manager.normalizeUri(row[0]))
    #
    # print(res)

    for k, v in contexts_list.items():
        # print (v)
        for cont in v:
            if subject in cont:
                if k not in contexts:
                    contexts.append(k)

    return contexts


# print(get_contexts(graph=g, subject="ns1:context_walking_home_boredom"))

unique_activities = list(activity_rename_mapping['extrasensory'].values()) + list(activity_rename_mapping['casas'].values())
unique_activities = pd.Series(unique_activities).drop_duplicates().values.tolist()
unique_sets = list(combinations(unique_activities, 2))
usets = []

for set_ in unique_sets:

    # activity_pattern = f'{set_[0]}_{set_[1]}'
    # print(activity_pattern)
    usets.append(f'{set_[0]}_{set_[1]}')
    usets.append(f'{set_[0]}+{set_[1]}')
total_sets = unique_activities + usets

# print(total_sets)

final_output = list()
for pattern in total_sets:
    search_value = "context_"+pattern.lower()
    # print(get_contexts(graph=g, subject="ns1:context_walking_home_boredom"))
    if (len(search_value.split("_"))>2):
        search_value_new = search_value.split("_")[0]+"_"+search_value.split("_")[1]
        # print(search_value_new)
        res = get_contexts(graph=g, subject=search_value_new)
        search_value_new = search_value.split("_")[0] + "_" + search_value.split("_")[2]
        # print(search_value_new)
        res += get_contexts(graph=g, subject=search_value_new)
    else:
        search_value_new = search_value.split("_")[0]+"_"+search_value.split("_")[1]
        # print(search_value_new)
        res = get_contexts(graph=g, subject=search_value_new)

    final_output += [search_value + "," + str(res)]
    # print(search_value, res)
    # if new_list:
print(final_output)
pd.Series(final_output).to_csv('onto_sets_final.csv',index=False,header=False)
