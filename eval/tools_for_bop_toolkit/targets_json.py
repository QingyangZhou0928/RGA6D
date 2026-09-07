import json
import os

dataset_root = './datasets/BOP/tless'
test_split = 'test_primesense' 
original_targets_path = os.path.join(dataset_root, 'test_targets_bop19.json')
custom_targets_path = os.path.join(dataset_root, 'targets_custom_1_30.json')

target_obj_ids = list(range(1, 31))

if os.path.exists(original_targets_path):
    with open(original_targets_path, 'r') as f:
        original_targets = json.load(f)
    custom_targets = []
    existing_scenes = [int(d) for d in os.listdir(os.path.join(dataset_root, test_split)) if d.isdigit()]

    for target in original_targets:
        if target['scene_id'] in existing_scenes and target['obj_id'] in target_obj_ids:
            custom_targets.append(target)
            
    with open(custom_targets_path, 'w') as f:
        json.dump(custom_targets, f, indent=2)
    
    print(f"success! contain {len(custom_targets)} targets for evaluation")
    print(f"new file: {custom_targets_path}")
else:
    print("ERROR")