import os
import json
import argparse
import numpy as np
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', '-e', required=True, type=str)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    with open(args.exp, 'r') as f:
        data = json.load(f)

    cam_extr = np.eye(3)
    points = (cam_extr @ np.array(data['pointclouds_up'], dtype=np.float32).transpose(1, 0)).transpose(1, 0)
    palm = cam_extr @ np.array(data['handpalm'], dtype=np.float32)
    trans = cam_extr @ np.array(data['objtrans'], dtype=np.float32)
    points = points + palm + trans
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.5)
    pcd = pcd.select_by_index(ind)

    mesh_path = args.exp.replace('json', 'obj')
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([mesh, pcd], window_name="Point Cloud and Mesh", width=800, height=600, mesh_show_back_face=True)

if __name__ == "__main__":
    main()
