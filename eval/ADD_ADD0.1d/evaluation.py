import numpy as np
import json
import os
from pathlib import Path
from scipy.spatial.distance import cdist
import open3d as o3d
import pandas as pd
import ast
from tqdm import tqdm
import gc

def load_point_cloud(ply_path):
    """loading PLY file"""
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    return points

def compute_model_diameter(model_pts):
    dist_matrix = cdist(model_pts, model_pts)
    diameter = np.max(dist_matrix)
    return diameter

def toOpen3dCloud(points,colors=None,normals=None):
    cloud = o3d.geometry.PointCloud()  
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64)) 
    if colors is not None:
        if colors.max()>1:
            colors = colors/255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64)) 
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud

def transform_points_einsum(model_pts, pose):
    points = np.asarray(model_pts)
    points_h = np.hstack([points, np.ones((len(points), 1))])
    transformed = np.einsum('ij,kj->ki', pose, points_h)
    
    pts = transformed[:, :3]
    pts = np.asarray(pts)
    return pts

def calculate_add_distance(model_pts, pose_gt, pose_pr):
    """
    compute ADD
    """
    pts_gt = transform_points_einsum(model_pts, pose_gt)
    pts_pr = transform_points_einsum(model_pts, pose_pr)
    
    add_distance = np.mean(np.linalg.norm(pts_pr - pts_gt, 2, 1))
    
    return add_distance

def calculate_add_s_distance(model_pts, pose_gt, pose_pr):
    """
    compute ADD-S
    """
    pts_gt = transform_points_einsum(model_pts, pose_gt)
    pts_pr = transform_points_einsum(model_pts, pose_pr)
    
    distances = np.min(cdist(pts_pr, pts_gt), axis=1)  
    add_s = np.mean(distances)
    
    return add_s

def find_intervals(series):
    diff = series != series.shift()
    groups = diff.cumsum()
    
    result = {}
    for group_id in groups.unique():
        mask = groups == group_id
        indices = series.index[mask].tolist()
        value = series.iloc[indices[0]]
        
        if value not in result:
            result[value] = []
        result[value].append([indices[0], indices[-1]])
    
    return result

if __name__ == "__main__":

    result_path = f'./results/predator_pose_estimation_results.xlsx'
    df = pd.read_excel(result_path)
    with open('./datasets/BOP/lm/models_eval/models_info.json', 'r') as f:
        data = json.load(f)
    with open('./datasets/BOP/lm/test_targets_bop19.json', 'r') as f:
        bop = json.load(f)
    result = []
    drop_indices = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="data"):
        if index==0:
            obj_id=row['object_id']
            model_path = f"./datasets/BOP/lm/models_eval/obj_{row['object_id']:06d}.ply"
            model_pts = load_point_cloud(model_path)  
            diameter = data[str(row['object_id'])]['diameter']
        
        if row['object_id']!=obj_id:
            if model_pts is not None: 
                del model_pts
                del diameter
            gc.collect()
            obj_id=row['object_id']
            model_path = f"./datasets/BOP/lm/models_eval/obj_{row['object_id']:06d}.ply"
            model_pts = load_point_cloud(model_path)  
            diameter = data[str(row['object_id'])]['diameter']

        flag = False
        for bop_ in bop:
            if bop_["im_id"]==row['frame_id'] and bop_["obj_id"]==row['object_id']:
                flag = True
                break
            if bop_["obj_id"]>row['object_id']:
                break
        if not flag:
            drop_indices.append(index)
            continue

        gt_pose = np.load(f"./datasets/lm_reconstruct/{row['object_id']:06d}/frame_{row['frame_id']}/T_obj2cam.npy")
        gt_pose = np.asarray(gt_pose)
        pred_pose = row['pred_transformation_obj_in_tgt']
        pred_pose = ast.literal_eval(pred_pose)       
        pred_pose = np.asarray(pred_pose)

        # ADD
        add = calculate_add_distance(model_pts, gt_pose, pred_pose)
        # ADD-S
        add_s = calculate_add_s_distance(model_pts, gt_pose, pred_pose)
        # ADD-0.1d
        add_01d_success = add < (diameter * 0.1)
        # ADD-S-0.1d
        add_s_01d_success = add_s < (diameter * 0.1)

        result.append({
        'add': add,
        'add_s': add_s,
        'add_01d_success': add_01d_success,
        'add_s_01d_success': add_s_01d_success,
        'model_diameter': diameter,
        })
    result = pd.DataFrame(result)
    df.drop(drop_indices,inplace=True)
    result.index = df.index
    df = pd.concat([df, result], axis=1)
    df.to_excel(f'./results/pose_estimation_results-add.xlsx', index=False, engine='openpyxl')

    results = []
    obj = find_intervals(df.loc[:,'object_id'])
    obj_id_list = list(obj.keys())

    for ob_id in  obj_id_list:
        start_index = obj[ob_id][0][0]
        end_index = obj[ob_id][0][1] 
        frame_num = end_index+1 -start_index
        add_mean = df.loc[start_index:end_index+1,'add'].mean()
        add_s_mean = df.loc[start_index:end_index+1,'add_s'].mean()
        add_01d_success = df.loc[start_index:end_index+1,'add_01d_success'].mean()
        add_s_01d_success = df.loc[start_index:end_index+1,'add_s_01d_success'].mean()
        results.append({
            'object_id': ob_id,
            'frame_num': frame_num,
            'add-mean': add_mean,
            'add_s-mean': add_s_mean,
            'add_01d_success': add_01d_success,
            'add_s_01d_success': add_s_01d_success,
            })

    df = pd.DataFrame(results)
    df.to_excel(f'./results/pose_estimation_results——统计.xlsx', index=False, engine='openpyxl')
