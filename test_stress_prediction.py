import requests
import json
import time
import numpy as np
import sys

port = 8080
request_url = '0.0.0.0'

print("Testing service on port:", port)
st_times = int(time.time())
timestamps = np.arange(st_times, st_times + (80 * 3600), 3600)

context_list = ['Amusement', 'Commuting', 'Exercising', 'HavingMeal', 'HouseWork',
                'HouseWork,Relaxing', 'InAMeeting', 'OfficeWork', 'PreparingMeal',
                'Relaxing', 'Sleeping', 'UsingBathroom']

contexts = []
for i in range(40*60):
    num_pars = np.random.randint(low=1, high=2)
    contexts_arr = np.random.choice(context_list, num_pars)
    contexts.append(','.join(contexts_arr))

st = time.time()
responses = []
for i in range(40*60):
    response = requests.get(
        f"http://{request_url}:{port}/wellness/stress?timestamp={timestamps[i]}&contexts={contexts[i]}")
    print(response.text)
    responses.append(response.text)
total_time = time.time() - st
response_dict = json.loads(response.text)
print(total_time)
