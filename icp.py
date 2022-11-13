import os
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg
from glob import glob
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import copy

def make_pc_dir():

    if not os.path.exists("point-cloud-data/seq-01"):
        os.makedirs("point-cloud-data/seq-01")

    for i in range(1000):
        if i<10:
            raw_c_path = f"frame-00000{i}.color.png"
            raw_d_path = f"frame-00000{i}.depth.png"
        elif 10<=i<100:
            raw_c_path = f"frame-0000{i}.color.png"
            raw_d_path = f"frame-0000{i}.depth.png"
        else: 
            raw_c_path = f"frame-000{i}.color.png"
            raw_d_path = f"frame-000{i}.depth.png"
        
        make_point_cloud(raw_c_path, raw_d_path)

def make_point_cloud(raw_c_path, raw_d_path):
    color_raw = o3d.io.read_image("7-scenes-heads/seq-01/"+raw_c_path)
    depth_raw = o3d.io.read_image("7-scenes-heads/seq-01/"+raw_d_path)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw)
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    file_name = raw_c_path[:12]
    o3d.io.write_point_cloud(f"point-cloud-data/seq-01/{file_name}.pcd", pcd)

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    # source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                    #   zoom=0.4459,
                                    #   front=[0.9288, -0.2951, -0.2242],
                                    #   lookat=[1.6784, 2.0612, 1.4451],
                                    #   up=[-0.3402, -0.9189, -0.1996]
                                    )
if __name__ == "__main__":
    # Run this once
    # make_pc_dir()
    source = o3d.io.read_point_cloud("point-cloud-data/seq-01/frame-000000.pcd")
    target = o3d.io.read_point_cloud("point-cloud-data/seq-01/frame-000111.pcd")
    threshold = 0.02
    trans_init = np.asarray([[1., 0., 0., 0.], 
                    [0., 1., 0., 0.], 
                    [0., 0., 1., 0.], 
                    [0., 0., 0., 1.]])
    # draw_registration_result(source, target, trans_init)

    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    # print("Apply point-to-plane ICP")
    # reg_p2l = o3d.pipelines.registration.registration_icp(
    #     source, target, threshold, trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
    # print(reg_p2l)
    # print("Transformation is:")
    # print(reg_p2l.transformation)
    # draw_registration_result(source, target, reg_p2l.transformation)
    # # draw_registration_result(source, target, reg_p2p.transformation)