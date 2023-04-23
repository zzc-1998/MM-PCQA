import open3d as o3d
import numpy as np
import os
import time
import random
import argparse
def visualize(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([pcd])

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, patch_size):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    npoint = int(N/patch_size) + 1
    if N < npoint:
        idxes = np.hstack((np.tile(np.arange(N), npoint//N), np.random.randint(N, size=npoint%N)))
        return point[idxes, :]

    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def knn_patch(pcd_name, patch_size = 2048):
    pcd = o3d.io.read_point_cloud(pcd_name)
    # nomalize pc and set up kdtree
    points = pc_normalize(np.array(pcd.points))
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    fps_point = farthest_point_sample(points,patch_size)
    
    
    point_size = fps_point.shape[0]


    patch_list = []

    for i in  range(point_size):
        [_,idx,dis] = kdtree.search_knn_vector_3d(fps_point[i],patch_size)
        #print(pc_normalize(np.asarray(point)[idx[1:], :]))
        patch_list.append(np.asarray(points)[idx[:], :]) 
    
    #visualize(all_point(np.array(patch_list)))
    #visualize(point)
    return np.array(patch_list)



def main(config):
    objs = os.walk(config.path)  
    for path,dir_list,file_list in objs:  
      for obj in file_list:  
        start = time.time()
        pcd_name = os.path.join(path, obj)
        npy_name = os.path.join(config.out_path,obj.split('.')[0] + '.npy')
        print(pcd_name)
        patch = knn_patch(pcd_name)
        np.save(npy_name,patch)
        end = time.time()
        print('Consuming seconds /s :' + str(end-start))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument('--path', type=str, default = '/DATA/zzc/3dqa/wpc2.0_ply/') #path to the file that contain .ply models
    parser.add_argument('--out_path', type=str, default = '/DATA/zzc/3dqa/wpc2.0/wpc2.0_patch_2048/') # path to the output patches
    config = parser.parse_args()

    main(config)