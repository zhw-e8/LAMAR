import os
import json
import evaluate
folders = [
    folder
    for folder in os.listdir(os.path.dirname(__file__))
    if (os.path.isdir(os.path.join(os.path.dirname(__file__), folder)) and ("__" not in folder))
]
with open(os.path.join(os.path.dirname(__file__),"loading_config.json"), "r") as file:
    config=json.load(file)

METRICS_INDEX = {}
for folder in folders:
    path=os.path.abspath(os.path.join(os.path.dirname(__file__), folder))
    METRICS_INDEX[folder]=evaluate.load(path,config_name=config[folder])
