import os
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
from random import randint
import sys
import cv2
import numpy as np
import open3d as o3d
import pygicp
import time

from open3d.cuda.pybind.core import float32
from scipy.spatial.transform import Rotation
import rerun as rr

sys.path.append(os.path.dirname(__file__))
from arguments import SLAMParameters
from utils.traj_utils import TrajManager
from gaussian_renderer import render, render_2, network_gui
from tqdm import tqdm
import copy
from torchvision import transforms


class Pipe():
    def __init__(self, convert_SHs_python, compute_cov3D_python, debug):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug


class Tracker(SLAMParameters):
    def __init__(self, slam):
        super().__init__()
        self.dataset_path = slam.dataset_path
        self.output_path = slam.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = slam.verbose
        self.keyframe_th = slam.keyframe_th
        self.knn_max_distance = slam.knn_max_distance
        self.overlapped_th = slam.overlapped_th
        self.overlapped_th2 = slam.overlapped_th2
        self.downsample_rate_tracking = slam.downsample_rate_tracking
        self.downsample_rate_mapping = slam.downsample_rate_mapping
        self.test = slam.test
        self.rerun_viewer = slam.rerun_viewer

        self.camera_parameters = slam.camera_parameters
        self.W = slam.W
        self.H = slam.H
        self.fx = slam.fx
        self.fy = slam.fy
        self.cx = slam.cx
        self.cy = slam.cy
        self.depth_scale = slam.depth_scale
        self.depth_trunc = slam.depth_trunc
        self.cam_intrinsic = np.array([[self.fx, 0., self.cx],
                                       [0., self.fy, self.cy],
                                       [0., 0., 1]])
        self.padding_H = 10
        self.padding_W = 10

        self.viewer_fps = slam.viewer_fps
        self.keyframe_freq = slam.keyframe_freq
        self.max_correspondence_distance = slam.max_correspondence_distance
        self.reg = pygicp.FastGICP()

        # Camera poses
        self.trajmanager = TrajManager(self.camera_parameters[8], self.dataset_path)
        self.poses = [self.trajmanager.gt_poses[0]]

        # Keyframes(added to map gaussians)
        self.last_t = time.time()
        self.iteration_images = 0
        self.end_trigger = False
        self.covisible_keyframes = []
        self.new_target_trigger = False

        self.cam_t = []
        self.cam_R = []
        self.points_cat = []
        self.colors_cat = []
        self.rots_cat = []
        self.scales_cat = []
        self.trackable_mask = []
        self.from_last_tracking_keyframe = 0
        self.from_last_mapping_keyframe = 0
        self.scene_extent = 2.5
        self.levels = 10
        self.pipe = Pipe(self.convert_SHs_python, self.compute_cov3D_python, self.debug)
        self.bg_color = [1, 1, 1] if self.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")


        self.downsample_idxs_tracking, self.x_pre_tracking, self.y_pre_tracking, self.tracking_mask = self.set_downsample_filter(
            self.downsample_rate_tracking)
        self.downsample_idxs_mapping, self.x_pre_mapping, self.y_pre_mapping, self.mask_mapping = self.set_downsample_filter(
            self.downsample_rate_mapping)

        if self.downsample_rate_mapping == 1:
            self.scale_ratio = 1
        else:
            self.scale_ratio = self.downsample_rate_mapping / 2


        # Share
        self.train_iter = 0
        self.mapping_losses = []
        self.new_keyframes = []
        self.gaussian_keyframe_idxs = []

        self.shared_cam = slam.shared_cam
        self.shared_new_points = slam.shared_new_points
        self.shared_new_gaussians = slam.shared_new_gaussians
        self.shared_target_gaussians = slam.shared_target_gaussians
        self.end_of_dataset = slam.end_of_dataset
        self.is_tracking_keyframe_shared = slam.is_tracking_keyframe_shared
        self.is_mapping_keyframe_shared = slam.is_mapping_keyframe_shared
        self.target_gaussians_ready = slam.target_gaussians_ready
        self.new_points_ready = slam.new_points_ready
        self.final_pose = slam.final_pose
        self.demo = slam.demo
        self.is_mapping_process_started = slam.is_mapping_process_started

    def run(self):
        self.tracking()

    def tracking(self):

        if self.rerun_viewer:
            rr.init("3dgsviewer")
            rr.connect()

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))




        self.rgb_images, self.depth_images = self.get_images(f"{self.dataset_path}/images")
        self.num_images = len(self.rgb_images)
        self.reg.set_max_correspondence_distance(self.max_correspondence_distance)
        self.reg.set_max_knn_distance(self.knn_max_distance)
        if_mapping_keyframe = False

        step_list, R_h, distance = self.prepare_filter_data()  # xys

        self.total_start_time = time.time()
        pbar = tqdm(total=self.num_images)

        for ii in range(self.num_images):
            current_image = self.rgb_images.pop(0)
            current_depth_image = self.depth_images.pop(0)

            freq = self.generate_frequency(current_image)
            current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
            opacity_mask = self.multiLayer_spectrogram(freq, distance, step_list, ii)  # [0, 0.5]

            points_mapping, colors_mapping, z_values_mapping, mapping_filter, zero_filter_mapping, tracking_points, tracking_filter, tracking_mask = \
                self.downsample_and_get_tracking_and_mapping_pointcloud(current_depth_image, current_image)

            opacities_mapping, scales_mapping, colors_mapping = self.initial_opacities_scales(z_values_mapping,
                                                                                              opacity_mask,
                                                                                              colors_mapping,self.downsample_idxs_mapping, zero_filter_mapping)

            zero_filter_mapping = zero_filter_mapping[0]
            # GICP
            if self.iteration_images == 0:
                current_pose = self.poses[-1]

                if self.rerun_viewer:
                    # rr.set_time_sequence("step", self.iteration_images)
                    rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                    rr.log(
                        "cam/current",
                        rr.Transform3D(translation=self.poses[-1][:3,3],
                                    rotation=rr.Quaternion(xyzw=(Rotation.from_matrix(self.poses[-1][:3,:3])).as_quat()))
                    )
                    rr.log(
                        "cam/current",
                        rr.Pinhole(
                            resolution=[self.W, self.H],
                            image_from_camera=self.cam_intrinsic,
                            camera_xyz=rr.ViewCoordinates.RDF,
                        )
                    )
                    rr.log(
                        "cam/current",
                        rr.Image(current_image)
                    )

                # Update Camera pose #
                current_pose = np.linalg.inv(current_pose)
                T = current_pose[:3, 3]
                R = current_pose[:3, :3].transpose()
                T_ = torch.tensor(T).float()
                R_ = torch.tensor(R).float()

                # transform current points
                points_mapping = (R_ @ (points_mapping.transpose(0,1))).transpose(0,1) - R_ @ T_
                points_mapping = torch.tensor(points_mapping)
                tracking_points = torch.tensor(tracking_points)
                tracking_points = (R_ @ (tracking_points.transpose(1, 0))).transpose(0, 1) - R_ @ T_
                # Set initial pointcloud to target points
                self.reg.set_input_target(tracking_points)

                num_trackable_points = tracking_filter.shape[0]
                input_filter = np.zeros(tracking_points.shape[0], dtype=np.int32)
                input_filter[(tracking_filter)] = [range(1, num_trackable_points + 1)]

                self.reg.set_target_filter(num_trackable_points, input_filter)
                self.reg.calculate_target_covariance_with_filter()

                rots_mapping = torch.tile(torch.tensor([1., 0., 0., 0.]), (points_mapping.shape[0], 1))  # xys

                self.shared_new_gaussians.input_values(points_mapping, colors_mapping,
                                                       rots_mapping, scales_mapping,
                                                       opacities_mapping,
                                                       z_values_mapping, torch.tensor(current_pose),
                                                       tracking_mask, zero_filter_mapping,
                                                       opacity_mask.reshape(-1))

                # Add first keyframe
                current_depth_image = current_depth_image.astype(np.float32) / self.depth_scale
                self.shared_cam.setup_cam(R, T, current_image, current_depth_image)
                self.shared_cam.cam_idx[0] = self.iteration_images

                self.is_tracking_keyframe_shared[0] = 1

                while self.demo[0]:
                    time.sleep(1e-15)
                    self.total_start_time = time.time()
                if self.rerun_viewer:
                    # rr.set_time_sequence("step", self.iteration_images)
                    rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                    rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(points_mapping, colors=colors_mapping, radii=0.02))
            else:
                self.reg.set_input_source(tracking_points)
                num_trackable_points = tracking_filter.shape[0]
                input_filter = np.zeros(tracking_points.shape[0], dtype=np.int32)
                input_filter[(tracking_filter)] = [range(1, num_trackable_points + 1)]
                self.reg.set_source_filter(num_trackable_points, input_filter)

                initial_pose = self.poses[-1]

                current_pose = self.reg.align(initial_pose)
                self.poses.append(current_pose)

                if self.rerun_viewer:
                    # rr.set_time_sequence("step", self.iteration_images)
                    rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                    rr.log(
                        "cam/current",
                        rr.Transform3D(translation=self.poses[-1][:3,3],
                                    rotation=rr.Quaternion(xyzw=(Rotation.from_matrix(self.poses[-1][:3,:3])).as_quat()))
                    )
                    rr.log(
                        "cam/current",
                        rr.Pinhole(
                            resolution=[self.W, self.H],
                            image_from_camera=self.cam_intrinsic,
                            camera_xyz=rr.ViewCoordinates.RDF,
                        )
                    )
                    rr.log(
                        "cam/current",
                        rr.Image(current_image)
                    )

                # Update Camera pose #
                current_pose = np.linalg.inv(current_pose)
                T = current_pose[:3, 3]
                R = current_pose[:3, :3].transpose()
                T_ = torch.tensor(T).float()
                R_ = torch.tensor(R).float()

                rots_mapping = torch.tile(torch.tensor([1., 0., 0., 0.]), (points_mapping.shape[0], 1))  # xys


                current_depth_image = current_depth_image.astype(np.float32) / self.depth_scale
                self.shared_cam.setup_cam(R, T, current_image, current_depth_image)
                self.shared_cam.cam_idx[0] = self.iteration_images
                self.shared_cam.current_pose = torch.tensor(current_pose, dtype=torch.float32).cuda()

                # transform current points
                points_mapping = (R_ @ (points_mapping.transpose(0, 1))).transpose(0, 1) - R_ @ T_
                points_mapping = torch.tensor(points_mapping)


                # Use only trackable points when tracking
                target_corres, distances = self.reg.get_source_correspondence()  # get associated points source points

                # Keyframe selection #
                # Tracking keyframe
                beyond = np.where(distances > self.overlapped_th)[0]
                len_corres = len(distances)-len(beyond)  # 5e-4 self.overlapped_th

                if (self.iteration_images >= self.num_images - 1 \
                        or len_corres / distances.shape[0] < self.keyframe_th):
                    if_tracking_keyframe = True
                    self.from_last_tracking_keyframe = 0
                    # print('ratio:{}'.format(len_corres / distances.shape[0]))
                else:
                    if_tracking_keyframe = False
                    self.from_last_tracking_keyframe += 1

                # Mapping keyframe
                if (self.from_last_tracking_keyframe) % self.keyframe_freq == 0:
                    if_mapping_keyframe = True
                else:
                    if_mapping_keyframe = False

                if if_tracking_keyframe:
                    while self.is_tracking_keyframe_shared[0] or self.is_mapping_keyframe_shared[0]:
                        time.sleep(1e-15)

                    R_d = Rotation.from_matrix(R)  # from camera R
                    R_d_q = R_d.as_quat()  # xyzw
                    rots_mapping = self.quaternion_multiply(torch.tensor(R_d_q), rots_mapping)

                    # Erase overlapped points from current pointcloud before adding to map gaussian #
                    # Using filter
                    overlapped_indices = self.eliminate_overlapped2(distances, self.overlapped_th2) # 5e-5 self.overlapped_th
                    filter_tmp = torch.where(tracking_mask)[0]
                    indices_to_disable = filter_tmp[overlapped_indices]
                    tracking_mask[indices_to_disable] = False

                    # Add new gaussians
                    self.shared_new_gaussians.input_values(points_mapping, colors_mapping,
                                                           rots_mapping, scales_mapping,
                                                           opacities_mapping,
                                                           z_values_mapping, torch.tensor(current_pose),
                                                           tracking_mask, zero_filter_mapping, opacity_mask.reshape(-1))

                    self.is_tracking_keyframe_shared[0] = 1

                    # Get new target point
                    while not self.target_gaussians_ready[0]:
                        time.sleep(1e-15)

                    target_points = self.shared_target_gaussians.get_values_np()
                    self.reg.set_input_target(target_points)
                    self.target_gaussians_ready[0] = 0

                    if self.rerun_viewer:
                        # rr.set_time_sequence("step", self.iteration_images)
                        rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                        rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(points_mapping, colors=colors_mapping, radii=0.01))

                elif if_mapping_keyframe:
                    while self.is_tracking_keyframe_shared[0] or self.is_mapping_keyframe_shared[0]:
                        time.sleep(1e-15)

                    R_d = Rotation.from_matrix(R)  # from camera R
                    R_d_q = R_d.as_quat()  # xyzw
                    rots_mapping = self.quaternion_multiply(torch.tensor(R_d_q), rots_mapping)

                    self.shared_new_gaussians.input_values(points_mapping, colors_mapping,
                                                           rots_mapping, scales_mapping,
                                                           opacities_mapping,
                                                           z_values_mapping, torch.tensor(current_pose),
                                                           tracking_mask, zero_filter_mapping,
                                                           opacity_mask.reshape(-1))

                    self.is_mapping_keyframe_shared[0] = 1
            pbar.update(1)

            self.iteration_images += 1
            # if self.iteration_images % 10 == 0:
            #     print(
            #         f"ATE RMSE: {self.evaluate_ate(self.trajmanager.gt_poses[:len(self.poses)], self.poses) * 100.:.2f}")

        # Tracking end
        pbar.close()
        self.final_pose[:, :, :] = torch.tensor(np.array(self.poses)).float()
        self.end_of_dataset[0] = 1

        print(f"System FPS: {1 / ((time.time() - self.total_start_time) / self.num_images):.2f}")
        print(f"ATE RMSE: {self.evaluate_ate(self.trajmanager.gt_poses, self.poses) * 100.:.2f}")

    def get_images(self, images_folder):
        rgb_images = []
        depth_images = []
        if self.trajmanager.which_dataset == "replica":
            image_files = os.listdir(images_folder)
            image_files = sorted(image_files.copy())
            for key in tqdm(image_files):
                image_name = key.split(".")[0]
                depth_image_name = f"depth{image_name[5:]}"

                rgb_image = cv2.imread(f"{self.dataset_path}/images/{image_name}.jpg")
                depth_image = np.array(o3d.io.read_image(f"{self.dataset_path}/depth_images/{depth_image_name}.png"))

                rgb_images.append(rgb_image)
                depth_images.append(depth_image)
            return rgb_images, depth_images
        elif self.trajmanager.which_dataset == "tum":
            for i in tqdm(range(len(self.trajmanager.color_paths))):
                rgb_image = cv2.imread(self.trajmanager.color_paths[i])
                depth_image = np.array(o3d.io.read_image(self.trajmanager.depth_paths[i]))
                rgb_images.append(rgb_image)
                depth_images.append(depth_image)
            return rgb_images, depth_images

    def run_viewer(self, lower_speed=True):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            if time.time() - self.last_t < 1 / self.viewer_fps and lower_speed:
                break
            try:
                net_image_bytes = None
                custom_cam, do_training, self.pipe.convert_SHs_python, self.pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)[
                        "render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())

                self.last_t = time.time()
                network_gui.send(net_image_bytes, self.dataset_path)
                if do_training and (not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

    def quaternion_multiply(self, q1, Q2):
        # q1*Q2
        x0, y0, z0, w0 = q1

        return torch.stack([w0 * Q2[:, 0] + x0 * Q2[:, 3] + y0 * Q2[:, 2] - z0 * Q2[:, 1],
                            w0 * Q2[:, 1] + y0 * Q2[:, 3] + z0 * Q2[:, 0] - x0 * Q2[:, 2],
                            w0 * Q2[:, 2] + z0 * Q2[:, 3] + x0 * Q2[:, 1] - y0 * Q2[:, 0],
                            w0 * Q2[:, 3] - x0 * Q2[:, 0] - y0 * Q2[:, 1] - z0 * Q2[:, 2]]).transpose(0, 1)

    def set_downsample_filter(self, downsample_scale):
        # Get sampling idxs
        sample_interval = downsample_scale
        if sample_interval == 1:
            h_val = torch.arange(0, int(self.H))
            w_val = torch.arange(0, int(self.W))
        else:
            h_val = torch.arange(0, int(self.H))[::sample_interval]
            w_val = torch.arange(0, int(self.W))[::sample_interval]
            if h_val[-1] != self.H - 1:
                h_val = torch.cat([h_val, torch.tensor([self.H - 1])])
            if w_val[-1] != self.W - 1:
                w_val = torch.cat([w_val, torch.tensor([self.W - 1])])

        h_val = h_val * self.W
        a, b = torch.meshgrid(h_val, w_val)
        # For tensor indexing, we need tuple
        pick_idxs = ((a + b).flatten(),)

        # Get u, v values
        v, u = torch.meshgrid(torch.arange(0, self.H), torch.arange(0, self.W))
        u = u.flatten()[pick_idxs]
        v = v.flatten()[pick_idxs]

        # Calculate xy values, not multiplied with z_values
        x_pre = (u - self.cx) / self.fx  # * z_values
        y_pre = (v - self.cy) / self.fy  # * z_values

        mask = np.zeros((self.H * self.W), dtype=bool)
        mask[pick_idxs] = True
        # mask_img = mask.reshape(self.H, self.W)

        return pick_idxs, x_pre, y_pre, mask

    def downsample_and_get_tracking_and_mapping_pointcloud(self, depth_img, rgb_img):
        colors = torch.from_numpy(rgb_img).reshape(-1, 3).float()[self.downsample_idxs_mapping] / 255
        z_values = torch.from_numpy(depth_img.astype(np.float32)).flatten()[self.downsample_idxs_mapping] / self.depth_scale
        zero_filter = torch.where(z_values != 0)
        z_values = z_values[zero_filter]

        # Trackable gaussians (will be used in tracking)
        x = self.x_pre_mapping[zero_filter] * z_values
        y = self.y_pre_mapping[zero_filter] * z_values
        points = torch.stack([x, y, z_values], dim=-1)
        colors = colors[zero_filter]

        tracking_mask = torch.tensor(self.tracking_mask[self.downsample_idxs_mapping][zero_filter])
        tracking_points = points[tracking_mask,:]
        z_values_tracking = z_values[tracking_mask]
        filter = torch.where(z_values <= self.depth_trunc)[0]
        mask_depth = z_values <= self.depth_trunc
        tracking_mask = tracking_mask & mask_depth

        tracking_filter = torch.where(z_values_tracking <= self.depth_trunc)[0]     # index based on tracking

        return points, colors, z_values, filter, zero_filter, tracking_points.numpy(), tracking_filter.numpy(), tracking_mask

    def initial_opacities_scales(self, z_values, opacity_mask, colors, downsample_idxs=None,
                                 zero_filter=None):
        opacities = torch.ones((z_values.shape[0], 1)).float() * 0.99

        if downsample_idxs == None:
            opacity_mask_ = copy.deepcopy(opacity_mask).flatten()
        else:
            opacity_mask_ = copy.deepcopy(opacity_mask).flatten()[downsample_idxs][zero_filter]

        arg_low = torch.argwhere(opacity_mask_ == 0)
        arg_high = torch.argwhere(opacity_mask_ == 1)
        opacity_mask_[arg_low] = 2 * self.scale_ratio
        opacity_mask_[arg_high] = 1 * self.scale_ratio  # 1
        scale_gaussian1 = (z_values / self.fx) * opacity_mask_  # * 2 ** (self.levels - scale_level)/2     # z_values / self.fx * 2 ** (self.levels - scale_level)/4
        scale_gaussian2 = (z_values / self.fy) * opacity_mask_  # * 2 ** (self.levels - scale_level)/2     # z_values / self.fy * 2 ** (self.levels - scale_level)/4
        scale_gaussian3 = 5e-6 * torch.ones_like(scale_gaussian2)
        scales = torch.stack([scale_gaussian1, scale_gaussian2, scale_gaussian3], axis=1)

        return opacities, scales, colors

    def eliminate_overlapped2(self, distances, threshold):

        new_p_indices = np.where(distances < threshold)  # 5e-5

        return new_p_indices

    def align(self, model, data):

        np.set_printoptions(precision=3, suppress=True)
        model_zerocentered = model - model.mean(1).reshape((3, -1))
        data_zerocentered = data - data.mean(1).reshape((3, -1))

        W = np.zeros((3, 3))
        for column in range(model.shape[1]):
            W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
        U, d, Vh = np.linalg.linalg.svd(W.transpose())
        S = np.matrix(np.identity(3))
        if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
            S[2, 2] = -1
        rot = U * S * Vh
        trans = data.mean(1).reshape((3, -1)) - rot * model.mean(1).reshape((3, -1))

        model_aligned = rot * model + trans
        alignment_error = model_aligned - data

        trans_error = np.sqrt(np.sum(np.multiply(
            alignment_error, alignment_error), 0)).A[0]

        return rot, trans, trans_error

    def evaluate_ate(self, gt_traj, est_traj):

        gt_traj_pts = [gt_traj[idx][:3, 3] for idx in range(len(gt_traj))]
        gt_traj_pts_arr = np.array(gt_traj_pts)
        gt_traj_pts_tensor = torch.tensor(gt_traj_pts_arr)
        gt_traj_pts = torch.stack(tuple(gt_traj_pts_tensor)).detach().cpu().numpy().T

        est_traj_pts = [est_traj[idx][:3, 3] for idx in range(len(est_traj))]
        est_traj_pts_arr = np.array(est_traj_pts)
        est_traj_pts_tensor = torch.tensor(est_traj_pts_arr)
        est_traj_pts = torch.stack(tuple(est_traj_pts_tensor)).detach().cpu().numpy().T

        _, _, trans_error = self.align(gt_traj_pts, est_traj_pts)

        avg_trans_error = trans_error.mean()

        return avg_trans_error

    # gaussian low-pass filter
    def gaussian_lp(self, Dis, D0):
        return torch.exp(-Dis ** 2 / (2 * D0 ** 2))

    # gaussian high-pass filter
    def gaussian_hp(self, Dis, D0):
        return 1 - torch.exp(-Dis ** 2 / (2 * D0 ** 2))

    # gaussian band-pass filter
    def gaussian_bp(self, Dis, W, C):
        return torch.exp(-((Dis ** 2 - C ** 2) / (Dis * W)) ** 2)



    def prepare_filter_data(self):
        start = 0
        step_list = np.ones(self.levels)
        step_norm = (np.min([self.H, self.W]) / 2 - start) / np.sum(step_list)
        step_list = (step_norm * np.array(step_list)).astype(np.uint16)

        center_h = int(self.H / 2 + self.padding_H)
        center_w = int(self.W / 2 + self.padding_W)

        x_grid, y_grid = torch.meshgrid(torch.arange(self.W + self.padding_W * 2).cuda().float(),
                                        torch.arange(self.H + self.padding_H * 2).cuda().float(),
                                        indexing='xy')

        distance = torch.sqrt((x_grid - center_w) ** 2 + (y_grid - center_h) ** 2)

        R_h = np.sum(step_list[:-2])
        return step_list, R_h, distance


    def generate_frequency(self, current_image):
        if self.padding_H != 0 or self.padding_H != 0:
            current_image = cv2.copyMakeBorder(current_image, self.padding_H, self.padding_H, self.padding_W,
                                               self.padding_W, cv2.BORDER_REFLECT_101)

        gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)  # RGB or BGR ?
        gray = self.clahe.apply(gray)
        gray = torch.tensor(gray).cuda()

        freq = torch.fft.fft2(gray, dim=(0, 1), norm='ortho')
        freq = torch.fft.fftshift(freq, dim=(0, 1))

        return freq

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def multiLayer_spectrogram(self, freq, distance, step_list, ii):
        R1 = 0
        highfreq_mask = np.zeros((self.H, self.W), dtype=np.uint8)

        for i in range(2):
            if i == 0:
                continue
            if i != 0:
                filter_g = self.gaussian_hp(Dis=distance, D0=np.sum(
                    step_list[:i]))
            else:
                R2 = R1 + step_list[i]
                filter_g = self.gaussian_lp(Dis=distance, D0=R2)
                R1 = R2


            freq_g = torch.fft.fftshift(freq * filter_g, dim=(0, 1))
            gray_g = torch.fft.ifft2(freq_g, dim=(0, 1), norm='ortho').cpu().numpy()
            if self.padding_H != 0 or self.padding_H != 0:
                gray_g = gray_g[self.padding_H:-self.padding_H, self.padding_W:-self.padding_W]
            gray_g = np.abs(gray_g)
            gray_g_m = gray_g - np.min(gray_g)
            gray_g = gray_g_m / np.max(gray_g_m)

            if i != 0:
                gray_g = 255 * gray_g
                gray_g = np.round(gray_g).astype(np.uint8)
                _, opacity_mask = cv2.threshold(gray_g, 0, 255, cv2.THRESH_TRIANGLE)
                opacity_mask = (opacity_mask / 255).astype(np.uint8)
                highfreq_mask = highfreq_mask | opacity_mask

        return torch.tensor(highfreq_mask)

    def get_depth_and_silhouette(self, pts_3D, w2c):
        """
        Function to compute depth and silhouette for each gaussian.
        These are evaluated at gaussian center.
        """
        # Depth of each gaussian center in camera frame
        pts4 = torch.cat((pts_3D, torch.ones_like(pts_3D[:, :1])), dim=-1)
        pts_in_cam = (w2c @ pts4.transpose(0, 1)).transpose(0, 1)
        depth_z = pts_in_cam[:, 2].unsqueeze(-1)  # [num_gaussians, 1]
        depth_z_sq = torch.square(depth_z)  # [num_gaussians, 1]

        # Depth and Silhouette
        depth_silhouette = torch.zeros((pts_3D.shape[0], 3)).cuda().float()
        depth_silhouette[:, 0] = depth_z.squeeze(-1)
        depth_silhouette[:, 1] = 1.0
        depth_silhouette[:, 2] = depth_z_sq.squeeze(-1)

        return depth_silhouette

    def matrix_to_quaternion(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert rotations given as rotation matrices to quaternions.

        Args:
            matrix: Rotation matrices as tensor of shape (..., 3, 3).

        Returns:
            quaternions with real part first, as tensor of shape (..., 4).
        Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
        """
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

        batch_dim = matrix.shape[:-2]
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
            matrix.reshape(batch_dim + (9,)), dim=-1
        )

        q_abs = self._sqrt_positive_part(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            )
        )

        # we produce the desired quaternion multiplied by each of r, i, j, k
        quat_by_rijk = torch.stack(
            [
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
            ],
            dim=-2,
        )

        # We floor here at 0.1 but the exact level is not important; if q_abs is small,
        # the candidate won't be picked.
        flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
        quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

        # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
        # forall i; we pick the best-conditioned one (with the largest denominator)

        return quat_candidates[
               F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
               ].reshape(batch_dim + (4,))

    def _sqrt_positive_part(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns torch.sqrt(torch.max(0, x))
        but with a zero subgradient where x is 0.
        Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
        """
        ret = torch.zeros_like(x)
        positive_mask = x > 0
        ret[positive_mask] = torch.sqrt(x[positive_mask])
        return ret

    def quat_mult(self, q1, q2):
        w1, x1, y1, z1 = q1.T
        w2, x2, y2, z2 = q2.T
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z]).T


