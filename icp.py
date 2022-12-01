import os
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg
from glob import glob
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import copy
from itertools import islice

# def make_pc_dir():

#     if not os.path.exists("point-cloud-data-redkitchen/seq-01"):
#         os.makedirs("point-cloud-data-redkitchen/seq-01")

#     for i in range(1000):
#         if i<10:
#             raw_c_path = f"frame-00000{i}.color.png"
#             raw_d_path = f"frame-00000{i}.depth.png"
#         elif 10<=i<100:
#             raw_c_path = f"frame-0000{i}.color.png"
#             raw_d_path = f"frame-0000{i}.depth.png"
#         else: 
#             raw_c_path = f"frame-000{i}.color.png"
#             raw_d_path = f"frame-000{i}.depth.png"
        
#         make_point_cloud(raw_c_path, raw_d_path)

# def make_point_cloud(raw_c_path, raw_d_path):
#     color_raw = o3d.io.read_image("7-scenes-redkitchen/seq-01/"+raw_c_path)
#     depth_raw = o3d.io.read_image("7-scenes-redkitchen/seq-01/"+raw_d_path)

#     rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
#         color_raw, depth_raw)
    
#     pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
#     rgbd_image,
#     o3d.camera.PinholeCameraIntrinsic(
#         o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
#     )

#     # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

#     file_name = raw_c_path[:12]
#     o3d.io.write_point_cloud(f"point-cloud-data-redkitchen/seq-01/{file_name}.pcd", pcd)
def make_data_dict():
    data_dict = []
    with open('7-scenes-redkitchen-evaluation/3dmatch.log', 'r') as f:
        while True:
            # read 5 lines at a time
            next_n_lines = list(islice(f, 5))
            if not next_n_lines:
                break
            trans_matrix = []
            for each_line in next_n_lines[1:]:
                string = each_line.split('\t')[:4]
                list_int = [eval(i) for i in string]
                trans_matrix.append(list_int)
            data_dict.append({"source": int(next_n_lines[0][0]), "target": int(next_n_lines[0].split(" ")[1]), "matrix": np.array(trans_matrix)})
    return data_dict

def draw_registration_result(source, target, transformation=None):
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
    
    data_dict = make_data_dict()[:100]
    icp_to_gt = []
    icp_to_init = []
    for i in range(len(data_dict)):

        source_idx = data_dict[i]['source']
        target_idx = data_dict[i]['target']
        gt_transformation = data_dict[i]['matrix']
        source = o3d.io.read_point_cloud(f"7-scenes-redkitchen/cloud_bin_{source_idx}.ply")
    #     source.paint_uniform_color([1, 0.706, 0])
        target = o3d.io.read_point_cloud(f"7-scenes-redkitchen/cloud_bin_{target_idx}.ply")
        gt_target = o3d.io.read_point_cloud(f"7-scenes-redkitchen/cloud_bin_{target_idx}.ply")
    #     target.paint_uniform_color([0,0,0.7])

        threshold = 0.02
        # print("Initial alignment")
        init_evaluation = o3d.pipelines.registration.evaluate_registration(
            source, target, threshold)
        gt_target = gt_target.transform(gt_transformation)

        # print("ground truth alignment")
        evaluation = o3d.pipelines.registration.evaluate_registration(
            source, gt_target, threshold)
        max_overlap = np.asarray(len(evaluation.correspondence_set))

        trans_init = np.asarray([[1., 0., 0., 0.], 
                        [0., 1., 0., 0.], 
                        [0., 0., 1., 0.], 
                        [0., 0., 0., 1.]])

        # print("icp alignment")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        evaluation = o3d.pipelines.registration.evaluate_registration(
            source, target.transform(reg_p2p.transformation), threshold)

        icp_overlap = np.asarray(len(evaluation.correspondence_set))/max_overlap
        from_initial = np.asarray(len(evaluation.correspondence_set))/np.asarray(len(init_evaluation.correspondence_set))

        icp_to_gt.append(icp_overlap)
        icp_to_init.append(from_initial)

        print(f"data {i} => icp/gt: {icp_overlap}, icp/initial: {from_initial}, gt: {max_overlap}, icp: {len(evaluation.correspondence_set)}, initial: {len(init_evaluation.correspondence_set)}")

    print("-------icp summary--------")
    print(f"icp/gt mean: {np.mean(icp_to_gt)}, icp/init mean: {np.mean(icp_to_init)}")

    # source = o3d.io.read_point_cloud(f"7-scenes-redkitchen/cloud_bin_0.ply")
  
    # source.paint_uniform_color([1, 0.706, 0])
    # target = o3d.io.read_point_cloud(f"7-scenes-redkitchen/cloud_bin_1.ply")

    # target.paint_uniform_color([0,0,0.7])

    # threshold = 0.02
    # init_evaluation = o3d.pipelines.registration.evaluate_registration(
    #         source, target, threshold)
    # print(np.asarray(len(init_evaluation.correspondence_set)))

    # trans_init = np.asarray([[1., 0., 0., 0.], 
    #                     [0., 1., 0., 0.], 
    #                     [0., 0., 1., 0.], 
    #                     [0., 0., 0., 1.]])

#     trans_init = np.asarray([[0.9969376500,	0.0591009984,	-0.0512093132,	-0.0955122912],
# [-0.0586364921,	0.9982238900,	0.0105274199	,-0.0318814123],
# [0.0517405408,	-0.0074924468,	0.9986324550,	0.1193341160],
# [0.0000000000,	0.0000000000	,0.0000000000	,1.0000000000]])

    # # print("icp alignment")
    # draw_registration_result(source, target, transformation=None)
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     source, target, threshold, trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # evaluation = o3d.pipelines.registration.evaluate_registration(
    #     source, target.transform(reg_p2p.transformation), threshold)
    # print(np.asarray(len(evaluation.correspondence_set)))

    # draw_registration_result(source, target)
