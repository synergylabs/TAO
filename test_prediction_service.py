import requests
import json
import time
import numpy as np
import sys

port = 8080
request_url = '0.0.0.0'

print("Testing service on port:", port)
st_times = int(time.time())
timestamps = np.arange(st_times,st_times+80,2)

# for i in range(40):
#     timestamps.append(int(time.time()))
#     time.sleep(1)
activity_list= ["Lying down","Sitting", "Walking", "Running", "Bicycling", "Sleeping", "Lab work", "In class",
              "In a meeting", "Drive - I'm the driver", "Drive - I'm a passenger", "Exercise", "Cooking",
              "Shopping", "Strolling", "Drinking (alcohol)","Bathing - shower", "Cleaning", "Doing laundry",
              "Washing dishes", "WatchingTV", "Surfing the internet", "Singing", "Talking", "Computer work",
              "Eating", "Toilet", "Grooming", "Dressing", "Stairs - going up", "Stairs - going down", "Standing",
              "With co-workers", "With friends"]
activities =[]
for i in range(40):
    num_pars = np.random.randint(low=1, high=4)
    activities_arr=np.random.choice(activity_list, num_pars)
    activities.append(','.join(activities_arr))


st = time.time()
responses = []
for i in range(40):
    response = requests.get(
        f"http://{request_url}:{port}/predict?timestamp={timestamps[i]}&activities={activities[i]}")
    responses.append(response.text)
total_time = time.time()-st
response_dict = json.loads(response.text)
print(total_time)