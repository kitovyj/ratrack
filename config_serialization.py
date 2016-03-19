import os.path
import json

from tracking import *

def load_tracking_config(file_name):    
    config = tracking.Tracking.Configuration()
    
    if not os.path.isfile(file_name):
        save_tacking_config(config, file_name)
    else:
        data_file = open(file_name)
        data = json.load(data_file)    
        config.skeletonization_res_width = data["skeletonization_res_width"]
        config.skeletonization_res_width = data["skeletonization_res_height"]
        config.vertebra_length = data["vertebra_length"]
        
    return config


def save_tacking_config(config, file_name):
    file = open(file_name, 'w') 
    json.dump({ 'skeletonization_res_width': config.skeletonization_res_width,
                'skeletonization_res_height': config.skeletonization_res_height,
                'vertebra_length': config.vertebra_length }, file, indent = 4)
