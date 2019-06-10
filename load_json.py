import json

def load_json(path_to_json):
    with open(path_to_json, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name