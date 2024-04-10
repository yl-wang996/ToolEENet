import os
from urdf_generation import URDFLoader
import subprocess
import time

script_name = '/homeL/1wang/workspace/toolee_ws/src/dataset_generation/src/pcd_render.py'


load = URDFLoader()
category_list = load.get_category_list()
for category in category_list:
    obj_list = load.get_obj_list(category)
    for obj in obj_list:
        command = [
            'python', script_name,
            '--headless', 'True',
            '--visualize', 'False',
            '--save', 'True',
            '--max_render_per_obj', '100',
            '--num_per_ins', '2',
            '--env_num', '10',
            '--total_num_per_obj', '1000',
            '--obj_name', obj,
            '--cat_name', category,
        ]
        # Run the command in a blocking way
        subprocess.run(command, check=True)
        time.sleep(1)



