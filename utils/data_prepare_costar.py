import pickle, yaml, os, sys
import numpy as np
from os.path import join, exists, dirname, abspath
from sklearn.neighbors import KDTree

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_tool import DataProcessing as DP

data_config = os.path.join(BASE_DIR, 'semantic-costar.yaml')
DATA = yaml.safe_load(open(data_config, 'r'))
remap_dict = DATA["learning_map"]
max_key = max(remap_dict.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)
remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

grid_size = 0.06
dataset_path = '/home/andimo/Desktop/costar/sequences'
output_path = '/home/andimo/Desktop/costar/sequences' + '_' + str(grid_size)
seq_list = np.sort(os.listdir(dataset_path))

for seq_id in seq_list:
    print('sequence ' + seq_id + ' start')
    seq_path = join(dataset_path, seq_id)
    seq_path_out = join(output_path, seq_id)
    h5py_path = join(seq_path, 'h5py')
    pc_path_out = join(seq_path_out, 'velodyne')
    KDTree_path_out = join(seq_path_out, 'KDTree')
    os.makedirs(seq_path_out) if not exists(seq_path_out) else None
    os.makedirs(pc_path_out) if not exists(pc_path_out) else None
    os.makedirs(KDTree_path_out) if not exists(KDTree_path_out) else None

    if int(seq_id) < 3:
        label_path_out = join(seq_path_out, 'labels')
        os.makedirs(label_path_out) if not exists(label_path_out) else None
        scan_list = np.sort(os.listdir(h5py_path))
        for scan_id in scan_list:
            print(scan_id)
            points = DP.load_pc_costar(join(h5py_path, scan_id))
            labels = DP.load_label_costar(join(h5py_path, scan_id))
            sub_points, sub_labels = DP.grid_sub_sampling(points, labels=labels, grid_size=grid_size)
            search_tree = KDTree(sub_points)
            KDTree_save = join(KDTree_path_out, str(scan_id[:-5]) + '.pkl')
            np.save(join(pc_path_out, scan_id)[:-5], sub_points)
            np.save(join(label_path_out, scan_id)[:-5], sub_labels)
            with open(KDTree_save, 'wb') as f:
                pickle.dump(search_tree, f)
            if seq_id == '02':
                proj_path = join(seq_path_out, 'proj')
                os.makedirs(proj_path) if not exists(proj_path) else None
                proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
                proj_inds = proj_inds.astype(np.int32)
                proj_save = join(proj_path, str(scan_id[:-5]) + '_proj.pkl')
                with open(proj_save, 'wb') as f:
                    pickle.dump([proj_inds], f)
    else:
        proj_path = join(seq_path_out, 'proj')
        os.makedirs(proj_path) if not exists(proj_path) else None
        scan_list = np.sort(os.listdir(h5py_path))
        for scan_id in scan_list:
            print(scan_id)
            points = DP.load_pc_costar(join(h5py_path, scan_id))
            sub_points = DP.grid_sub_sampling(points, grid_size=0.06)
            search_tree = KDTree(sub_points)
            proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            KDTree_save = join(KDTree_path_out, str(scan_id[:-5]) + '.pkl')
            proj_save = join(proj_path, str(scan_id[:-5]) + '_proj.pkl')
            np.save(join(pc_path_out, scan_id)[:-5], sub_points)