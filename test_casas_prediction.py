import requests
import json
import time
import numpy as np
import sys

port = 8080
request_url = 'edusense-compute-5.andrew.cmu.edu'

print("Testing service on port:", port)
st_times = int(time.time())
timestamps = np.arange(st_times, st_times + (80 * 60), 60)

activity_list = [
    "step_out",
    "toilet",
    "onphone",
    "grooming",
    "step_in",
    "lying",
    "drinking",
    "watching_tv",
    "dressing_up",
    "taking_meds",
    "wakingup",
    "reading",
    "cooking",
    "eating",
    "shower",
    "dishes_home",
    "sleeping",
    "office_work",
    "meeting_friends",
    "exercising",
    "laundry_home"]

activities = []
for i in range(40):
    num_pars = np.random.randint(low=1, high=2)
    activities_arr = np.random.choice(activity_list, num_pars)
    activities.append(','.join(activities_arr))

st = time.time()
responses = []
for i in range(40):
    response = requests.get(
        f"http://{request_url}:{port}/casas?timestamp={timestamps[i]}&activities={activities[i]}")
    print(response.text)
    responses.append(response.text)
total_time = time.time() - st
response_dict = json.loads(response.text)
print(total_time)
