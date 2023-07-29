# TAO: Context Detection from Daily Activity Patterns using Temporal Analysis and Ontology

[[paper (IMWUT 2023)](https://doi.org/10.1145/3610896)]
[[talk (IMWUT 2023)](https://www.youtube.com/)]
[[demo video](https://www.youtube.com/)]

**Authors:**
[[Sudershan Boovaraghavan](https://sudershanb.com/)]
[[Prasoon Patidar](http://prasoonpatidar.com/)]
[[Yuvraj Agarwal](https://www.synergylabs.org/yuvraj/)]

**Abstract:**
Translating fine-grained activity detection (e.g., phone ring, talking interspersed 
with silence and walking) into semantically meaningful and richer contextual information 
(e.g., on a phone call for 20 minutes while exercising) is essential towards enabling a range 
of healthcare and human-computer interaction applications. Prior work has proposed building 
ontologies or temporal analysis of activity patterns with limited success in capturing complex 
real-world context patterns. We present TAO, a hybrid system that leverages OWL-based ontologies 
and temporal clustering approaches to detect high-level contexts from human activities. TAO can 
characterize sequential activities that happen one after the other and activities that are 
interleaved or occur in parallel to detect a richer set of contexts more accurately than 
prior work. We evaluate TAO on real-world activity datasets (Casas and Extrasensory) 
and show that our system achieves, on average, 87% and 80% accuracy for context detection, 
respectively. We deploy and evaluate TAO in a real-world setting with eight participants 
using our system for three hours each, demonstrating TAO’s ability to capture semantically 
meaningful contexts in the real world. Finally, to showcase the usefulness of contexts, 
we prototype wellness applications that assess productivity and stress and show that 
the wellness metrics calculated using contexts provided by TAO are much closer to the 
ground truth (on average within 1.1%), as compared to the baseline approach (on average within 30%).
## Reference

[Download Paper Here](https://doi.org/10.1145/3610896)


BibTeX Reference:

```
@inproceedings{10.1145/
}
```


## Installation:

### 1. Clone (or Fork!) this repository
```
git clone git@github.com:synergylabs/TAO.git
```

### 2. Create a virtual environment and install python packages
We recommend using conda. Tested on `Ubuntu 20.04`, with `python 3.7`.

```bash
conda create -n "tao" python=3.7
conda activate tao
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch

python -m pip install -r requirements.txt
```

## Usage:

### 1. Using activity log files:

Save the activity log files in the folder: ```data_collection_pipeline/real_gt_data/```

```bash
cd ontological-Pipeline/

streamlit run run_onto_pipeline.py
```

### 2. Using REST API:

```bash

streamlit run API_server.py
```

#### Example usage:
```python
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

activities =[]

responses = []
for i in range(40):
    response = requests.get(
        f"http://{request_url}:{port}/predict?timestamp={timestamps[i]}&activities={activities[i]}")
    responses.append(response.text)
response_dict = json.loads(response.text)
```

[//]: # (## Directory structure:)



## References:


## Contact:

