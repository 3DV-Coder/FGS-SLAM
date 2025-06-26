import os
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
import copy
import random
import sys
import cv2
import numpy as np
import time
import rerun as rr
sys.path.append(os.path.dirname(__file__))
from arguments import SLAMParameters
from utils.traj_utils import TrajManager
from utils.loss_utils import l1_loss, ssim, frequency
from scene import GaussianModel
from gaussian_renderer import render, render_3, network_gui
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import open3d as o3d
import matplotlib.pyplot as plt

class Pipe():
    def __init__(self, convert_SHs_python, compute_cov3D_python, debug):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug

class Mapper(SLAMParameters):
    def __init__(self, slam):
        super().__init__()
        self.dataset_path = slam.dataset_path
        self.output_path = slam.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = slam.verbose
        self.keyframe_th = float(slam.keyframe_th)
        self.trackable_opacity_th = slam.trackable_opacity_th
        self.save_results = slam.save_results
        self.rerun_viewer = slam.rerun_viewer
        self.iter_shared = slam.iter_shared

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
                                       [0.,0.,1]])

        self.downsample_rate_tracking = slam.downsample_rate_tracking
        self.downsample_rate_mapping = slam.downsample_rate_mapping
        self.viewer_fps = slam.viewer_fps
        self.keyframe_freq = slam.keyframe_freq

        # Camera poses
        self.trajmanager = TrajManager(self.camera_parameters[8], self.dataset_path)
        self.poses = [torch.tensor(self.trajmanager.gt_poses[0], dtype=torch.float32).cuda()]
        # Keyframes(added to map gaussians)
        self.keyframe_idxs = []
        self.last_t = time.time()
        self.iteration_images = 0
        self.end_trigger = False
        self.covisible_keyframes = []
        self.new_target_trigger = False
        self.start_trigger = False
        self.if_mapping_keyframe = False
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
        if self.trajmanager.which_dataset == "replica":
            self.prune_th = 1
        else:
            self.prune_th = 10.0

        self.levels = 2
        self.prune_num = 0

        # self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(self.downsample_rate)
        self.downsample_idxs_tracking, self.x_pre_tracking, self.y_pre_tracking, self.tracking_down_mask = self.set_downsample_filter(self.downsample_rate_tracking)
        self.tracking_down_mask = torch.tensor(self.tracking_down_mask).cuda()
        self.downsample_idxs_mapping, self.x_pre_mapping, self.y_pre_mapping, self.mapping_down_mask = self.set_downsample_filter(self.downsample_rate_mapping)

        if self.downsample_rate_mapping == 1:
            self.high_freq_mask = torch.tensor(self.set_downsample_mask(2)).cuda()  # 2
            self.low_freq_mask = torch.tensor(self.set_downsample_mask(4)).cuda()  # 4
        else:
            self.high_freq_mask = torch.tensor(self.set_downsample_mask(self.downsample_rate_mapping)).cuda()  # 2
            self.low_freq_mask = torch.tensor(self.set_downsample_mask(self.downsample_rate_mapping*2)).cuda()   # 4

        self.gaussians = GaussianModel(self.sh_degree)
        self.pipe = Pipe(self.convert_SHs_python, self.compute_cov3D_python, self.debug)
        self.bg_color = [1, 1, 1] if self.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        self.train_iter = 0
        self.mapping_cams = []
        self.visibility_filters = []
        self.tracking_id_list = []
        self.mapping_id_list = []
        self.train_mapping_list = [-1]
        self.covisible_keyframes_group = {}
        self.mapping_losses = []
        self.gaussian_keyframe_idxs = []

        self.shared_cam = slam.shared_cam
        self.shared_new_points = slam.shared_new_points
        self.shared_new_gaussians = slam.shared_new_gaussians
        self.shared_target_gaussians = slam.shared_target_gaussians
        self.end_of_dataset = slam.end_of_dataset
        self.is_tracking_keyframe_shared = slam.is_tracking_keyframe_shared
        self.is_mapping_keyframe_shared = slam.is_mapping_keyframe_shared
        self.is_silhouette_frame_shared = slam.is_silhouette_frame_shared

        self.target_gaussians_ready = slam.target_gaussians_ready
        self.final_pose = slam.final_pose
        self.demo = slam.demo
        self.is_mapping_process_started = slam.is_mapping_process_started

    def run(self):
        self.mapping()

    def mapping(self):
        if self.verbose:
            network_gui.init("127.0.0.1", 6009)

        if self.rerun_viewer:
            rr.init("3dgsviewer")
            rr.connect()

        # Mapping Process is ready to receive first frame
        self.is_mapping_process_started[0] = 1

        # Wait for initial gaussians
        while not self.is_tracking_keyframe_shared[0]:
            time.sleep(1e-15)

        self.total_start_time_viewer = time.time()

        points, colors, rots, scales, opacities, z_values, current_pose, tracking_mask, zero_filter, opacity_mask = self.shared_new_gaussians.get_values()

        high_freq_mask = self.high_freq_mask & opacity_mask
        low_freq_mask = self.low_freq_mask & (~opacity_mask)
        select_mask = high_freq_mask | low_freq_mask | self.tracking_down_mask

        select_mask = select_mask[self.downsample_idxs_mapping][zero_filter].bool()
        points = points[select_mask, :]
        colors = colors[select_mask, :]
        rots = rots[select_mask, :]
        scales = scales[select_mask, :]
        opacities = opacities[select_mask, :]
        z_values = z_values[select_mask]
        tracking_mask = tracking_mask[select_mask]
        tracking_filter = torch.where(tracking_mask)[0]

        self.gaussians.create_from_pcd2_tensor(points, colors, rots, scales, opacities, z_values, tracking_filter)
        self.gaussians.spatial_lr_scale = self.scene_extent
        self.gaussians.training_setup(self)
        self.gaussians.update_learning_rate(1)
        self.gaussians.active_sh_degree = self.gaussians.max_sh_degree
        self.is_tracking_keyframe_shared[0] = 0

        if self.demo[0]:
            a = time.time()
            while (time.time()-a)<30.:
                print(30.-(time.time()-a))
                self.run_viewer()
        self.demo[0] = 0
        self.ii = 0

        newcam = copy.deepcopy(self.shared_cam)
        newcam.on_cuda()

        render_pkg = render_3(newcam, self.gaussians, self.pipe, self.background, training_stage=self.training_stage)
        visibility_filter = render_pkg["visibility_filter"]
        self.visibility_filters.append(visibility_filter)
        self.mapping_cams.append(newcam)
        self.tracking_id_list.append(len(self.mapping_cams)-1)
        self.covisible_keyframes_group["group{:d}".format(len(self.tracking_id_list) - 1)] = []
        self.covisible_keyframes_group["group{:d}".format(len(self.tracking_id_list) - 1)].append(self.tracking_id_list[-1])
        
        self.keyframe_idxs.append(newcam.cam_idx[0])
        self.optimizing_num = 0

        while True:
            if self.end_of_dataset[0]:
                break

            if self.verbose:
                self.run_viewer()

            if self.is_tracking_keyframe_shared[0]:
                print('tracking')
                # Add new keyframe
                newcam = copy.deepcopy(self.shared_cam)
                newcam.on_cuda()
                self.mapping_cams.append(newcam)
                self.keyframe_idxs.append(newcam.cam_idx[0])

                self.tracking_id_list.append(len(self.mapping_cams)-1)
                self.covisible_keyframes_group["group{:d}".format(len(self.tracking_id_list) - 1)] = []
                self.covisible_keyframes_group["group{:d}".format(len(self.tracking_id_list) - 1)].append(self.tracking_id_list[-1])

                viewpoint_cam = newcam
                points, colors, rots, scales, _, z_values, _, tracking_filter, render_pkg = self.prepare_add_gaussian_mask(viewpoint_cam, tracking=True)
                visibility_filter = render_pkg["visibility_filter"]
                self.visibility_filters.append(visibility_filter)

                # Add new gaussians to map gaussians
                self.gaussians.add_from_pcd2_tensor_tracking(points, colors, rots, scales, z_values, tracking_filter)  # 合并新的高斯作为待优化参数

                self.get_loss(viewpoint_cam, render_pkg, new_keyframe=False)

                # Allocate new target points to shared memory
                target_points, target_rots, target_scales, target_opacity = self.gaussians.get_trackable_gaussians_tensor(self.trackable_opacity_th)
                self.shared_target_gaussians.input_values(target_points)
                self.target_gaussians_ready[0] = 1

                gaussian_num = self.gaussians.get_xyz.shape[0]
                print('gaussian_num:{}'.format(gaussian_num))
                self.is_tracking_keyframe_shared[0] = 0

            elif self.is_mapping_keyframe_shared[0]:
                # Add new keyframe
                newcam = copy.deepcopy(self.shared_cam)
                newcam.on_cuda()
                self.mapping_cams.append(newcam)
                self.keyframe_idxs.append(newcam.cam_idx[0])
                self.mapping_id_list.append(len(self.mapping_cams) - 1)

                viewpoint_cam = newcam
                points, colors, rots, scales, opacities, z_values, current_pose, tracking_filter, render_pkg = self.prepare_add_gaussian_mask(viewpoint_cam, tracking=False)

                # Add new gaussians to map gaussians
                self.gaussians.add_from_pcd2_tensor(points, colors, rots, scales, opacities, z_values, [])  # 合并新的高斯作为待优化参数

                self.get_loss(viewpoint_cam, render_pkg, new_keyframe=False)

                gaussian_num = self.gaussians.get_xyz.shape[0]
                print('gaussian_num:{}'.format(gaussian_num))
                self.is_mapping_keyframe_shared[0] = 0

            if len(self.mapping_cams)>0:
                if self.optimizing_num > 1000:
                    continue
                # train once on new keyframe, and random
                if len(self.mapping_id_list) > 0:
                    self.optimizing_num = 0
                    train_idx = self.mapping_id_list.pop(0)
                    viewpoint_cam = self.mapping_cams[train_idx]
                    new_keyframe = True

                    self.train_mapping_list = []
                    self.random_mapping_list = []
                    render_pkg = render_3(viewpoint_cam, self.gaussians, self.pipe, self.background, training_stage=self.training_stage)
                    visibility_filter = render_pkg["visibility_filter"]
                    for i in range(len(self.visibility_filters)):
                        previous_visibility_filter = self.visibility_filters[i]
                        gaussians_num = len(previous_visibility_filter)
                        co_visibility_num = torch.sum(
                            previous_visibility_filter & visibility_filter[:gaussians_num])
                        all_visibility_num = torch.sum(visibility_filter[:gaussians_num])
                        if co_visibility_num / all_visibility_num > 0.9:
                            self.covisible_keyframes_group["group{:d}".format(i)].append(train_idx)
                            self.train_mapping_list += self.covisible_keyframes_group["group{:d}".format(i)]
                        else:
                            self.random_mapping_list+= self.covisible_keyframes_group["group{:d}".format(i)]
                    self.train_mapping_list = set(self.train_mapping_list)
                    self.random_mapping_list = set(self.random_mapping_list)
                    self.random_mapping_list = list(self.random_mapping_list - self.train_mapping_list)
                    self.train_mapping_list = list(self.train_mapping_list)

                    slip_train_num = int(80 * 0.7)
                    slip_random_num = 80 - slip_train_num
                    if len(self.train_mapping_list) > 0:
                        random_num = min([len(self.train_mapping_list), slip_train_num])
                        self.train_mapping_list = random.sample(self.train_mapping_list, random_num)
                    else:
                        self.train_mapping_list = [-1]
                    if len(self.random_mapping_list) > 0:
                        random_num = min([len(self.random_mapping_list), slip_random_num])
                        ran_extend = random.sample(self.random_mapping_list, random_num)
                        self.train_mapping_list = self.train_mapping_list + ran_extend
                else:
                    self.optimizing_num += 1
                    train_idx = random.choice(self.train_mapping_list)
                    viewpoint_cam = self.mapping_cams[train_idx]
                    render_pkg = render_3(viewpoint_cam, self.gaussians, self.pipe, self.background, training_stage=self.training_stage)
                    new_keyframe = False

                self.get_loss(viewpoint_cam, render_pkg, new_keyframe)
        if self.verbose:
            while True:
                self.run_viewer(False)

        # End of data
        if self.save_results and not self.rerun_viewer:
            self.gaussians.save_ply(os.path.join(self.output_path, "scene.ply"))

        self.calc_2d_metric()

    def run_viewer(self, lower_speed=True):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            if time.time()-self.last_t < 1/self.viewer_fps and lower_speed:
                break
            try:
                net_image_bytes = None
                custom_cam, do_training, self.pipe.convert_SHs_python, self.pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer, view=False)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

                self.last_t = time.time()
                network_gui.send(net_image_bytes, self.dataset_path)
                if do_training and (not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

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


    def get_image_dirs(self, images_folder):
        color_paths = []
        depth_paths = []
        if self.trajmanager.which_dataset == "replica":
            images_folder = os.path.join(images_folder, "images")
            image_files = os.listdir(images_folder)
            image_files = sorted(image_files.copy())
            for key in tqdm(image_files):
                image_name = key.split(".")[0]
                depth_image_name = f"depth{image_name[5:]}"
                color_paths.append(f"{self.dataset_path}/images/{image_name}.jpg")
                depth_paths.append(f"{self.dataset_path}/depth_images/{depth_image_name}.png")

            return color_paths, depth_paths
        elif self.trajmanager.which_dataset == "tum":
            return self.trajmanager.color_paths, self.trajmanager.depth_paths


    def calc_2d_metric(self):
        psnrs = []
        ssims = []
        lpips = []

        cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to("cuda")
        original_resolution = True
        image_names, depth_image_names = self.get_image_dirs(self.dataset_path)
        final_poses = self.final_pose
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        with torch.no_grad():
            for i in tqdm(range(len(image_names))):
                cam = self.mapping_cams[0]
                c2w = final_poses[i]

                if original_resolution:
                    gt_rgb = cv2.imread(image_names[i])
                    gt_depth = cv2.imread(depth_image_names[i] ,cv2.IMREAD_UNCHANGED).astype(np.float32)

                    gt_rgb = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR)
                    gt_rgb = gt_rgb/255
                    gt_rgb_ = torch.from_numpy(gt_rgb).float().cuda().permute(2,0,1)

                    gt_depth_ = torch.from_numpy(gt_depth).float().cuda().unsqueeze(0)
                else:
                    gt_rgb_ = cam.original_image.cuda()
                    gt_rgb = np.asarray(gt_rgb_.detach().cpu()).squeeze().transpose((1,2,0))
                    gt_depth_ = cam.original_depth_image.cuda()

                w2c = np.linalg.inv(c2w)
                # rendered
                R = w2c[:3,:3].transpose()
                T = w2c[:3,3]

                cam.R = torch.tensor(R)
                cam.t = torch.tensor(T)
                if original_resolution:
                    cam.image_width = gt_rgb_.shape[2]
                    cam.image_height = gt_rgb_.shape[1]
                else:
                    pass

                cam.update_matrix()
                # rendered rgb
                ours_rgb_ = render(cam, self.gaussians, self.pipe, self.background)["render"]
                ours_rgb_ = torch.clamp(ours_rgb_, 0., 1.).cuda()

                valid_depth_mask_ = (gt_depth_>0)

                gt_rgb_ = gt_rgb_ * valid_depth_mask_
                ours_rgb_mask = ours_rgb_ * valid_depth_mask_

                square_error = (gt_rgb_-ours_rgb_mask)**2
                mse_error = torch.mean(torch.mean(square_error, axis=2))
                psnr = mse2psnr(mse_error)

                psnrs += [psnr.detach().cpu()]
                _, ssim_error = ssim(ours_rgb_mask, gt_rgb_)
                ssims += [ssim_error.detach().cpu()]
                lpips_value = cal_lpips(gt_rgb_.unsqueeze(0), ours_rgb_mask.unsqueeze(0))
                lpips += [lpips_value.detach().cpu()]

                image_render = np.clip(np.asarray(ours_rgb_.detach().cpu()).squeeze().transpose((1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                cv2.imshow('render_result', image_render[..., ::-1])
                cv2.waitKey(1)

                if self.save_results and ((i + 1) % 10 == 0 or i == len(image_names) - 1):
                    if not os.path.exists(f"{self.output_path}/render"):
                        os.mkdir(f"{self.output_path}/render")
                    cv2.imwrite(f"{self.output_path}/render/render_{i}.jpg", image_render[..., ::-1])

                if self.save_results and ((i+1)%100==0 or i==len(image_names)-1):
                    ours_rgb = np.asarray(ours_rgb_.detach().cpu()).squeeze().transpose((1,2,0))

                    axs[0].set_title("gt rgb")
                    axs[0].imshow(gt_rgb)
                    axs[0].axis("off")
                    axs[1].set_title("rendered rgb")
                    axs[1].imshow(ours_rgb)
                    axs[1].axis("off")
                    plt.suptitle(f'{i+1} frame')
                    plt.pause(1e-15)
                    plt.savefig(f"{self.output_path}/result_{i}.png")
                    plt.cla()

                torch.cuda.empty_cache()

            psnrs = np.array(psnrs)
            ssims = np.array(ssims)
            lpips = np.array(lpips)

            print(f"PSNR: {psnrs.mean():.2f}\nSSIM: {ssims.mean():.3f}\nLPIPS: {lpips.mean():.3f}")

    def check_silhouette(self, viewpoint_cam):
        # with torch.no_grad():
        params = {'means3D': self.gaussians.get_xyz, 'norm_rotations': self.gaussians.get_rotation,
                  'logit_opacities': self.gaussians.get_opacity, 'log_scales': self.gaussians.get_scaling}
        transformed_gaussians = {'means3D': params['means3D'],
                                 'norm_rotations': params['norm_rotations']}
        depth_sil_rendervar = self.transformed_params2depthplussilhouette(params, viewpoint_cam.current_pose,
                                                                          transformed_gaussians)
        render_pkg_silhouette = render_3(viewpoint_cam, self.gaussians, self.pipe, self.background,
                                         training_stage=self.training_stage,
                                         depth_sil_rendervar=depth_sil_rendervar)  # 光栅化渲染
        depth_image_sil = render_pkg_silhouette["render"]  # 渲染sil深度图
        depth = depth_image_sil[0, :, :].unsqueeze(0)
        silhouette = depth_image_sil[1, :, :]
        depth_sq = depth_image_sil[2, :, :].unsqueeze(0)

        return silhouette, depth, depth_sq

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

    def transformed_params2depthplussilhouette(self, params, w2c, transformed_gaussians):
        # Check if Gaussians are Isotropic
        if params['log_scales'].shape[1] == 1:
            log_scales = torch.tile(params['log_scales'], (1, 3))
        else:
            log_scales = params['log_scales']
        # Initialize Render Variables
        rendervar = {
            'means3D': transformed_gaussians['means3D'],
            'colors_precomp': self.get_depth_and_silhouette(transformed_gaussians['means3D'], w2c),
            'rotations': transformed_gaussians['norm_rotations'],
            'opacities': params['logit_opacities'],
            'scales': log_scales,
            'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
        }
        return rendervar

    def set_downsample_mask(self, downsample_scale):
        # Get sampling idxs
        sample_interval = downsample_scale
        if sample_interval == 1:
            h_val = torch.arange(0, int(self.H))
            w_val = torch.arange(0, int(self.W))
        else:
            h_val = torch.arange(0, int(self.H))[::sample_interval]
            w_val = torch.arange(0, int(self.W))[::sample_interval]
            if h_val[-1] != self.H -1:
                h_val = torch.cat([h_val, torch.tensor([self.H -1])])
            if w_val[-1] != self.W -1:
                w_val = torch.cat([w_val, torch.tensor([self.W -1])])
        h_val = h_val * self.W
        a, b = torch.meshgrid(h_val, w_val)
        # For tensor indexing, we need tuple
        pick_idxs = ((a + b).flatten(),)


        mask = np.zeros((self.H * self.W), dtype=bool)
        mask[pick_idxs] = True

        return mask

    def get_loss(self, viewpoint_cam, render_pkg, new_keyframe):
        if self.training_stage == 0:
            gt_image = viewpoint_cam.original_image.cuda()
            gt_depth_image = viewpoint_cam.original_depth_image.cuda()
        elif self.training_stage == 1:
            gt_image = viewpoint_cam.rgb_level_1.cuda()
            gt_depth_image = viewpoint_cam.depth_level_1.cuda()
        elif self.training_stage == 2:
            gt_image = viewpoint_cam.rgb_level_2.cuda()
            gt_depth_image = viewpoint_cam.depth_level_2.cuda()

        self.training = True

        depth_image = render_pkg["render_depth"]
        image = render_pkg["render"]

        mask = (gt_depth_image > 0.)
        mask = mask.detach()
        gt_image_ = gt_image * mask
        image_ = image * mask

        # Loss
        Ll1_map, Ll1 = l1_loss(image_, gt_image_)
        L_ssim_map, L_ssim = ssim(image_, gt_image_)

        d_max = 10.
        Ll1_d_map, Ll1_d = l1_loss(depth_image / d_max, gt_depth_image / d_max)

        loss_rgb = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - L_ssim)
        loss_d = Ll1_d

        scales = self.gaussians.get_scaling
        loss_isotropic = torch.abs(scales[:, :2] - scales[:, :2].mean(dim=1).view(-1, 1))
        loss_isotropic = loss_isotropic.mean() + 1*torch.abs(scales[:,-1] - 5e-6).mean()

        loss = loss_rgb + 0.1 * loss_d + 10*loss_isotropic

        loss.backward()
        with torch.no_grad():
            if self.train_iter % 200 == 0:  # 200
                valid_mask = self.gaussians.prune_large_and_transparent(0.005, self.prune_th)
                self.prune_num += torch.sum(~valid_mask)
                print(' count pruning num:{}'.format(self.prune_num))
                for i in range(len(self.visibility_filters)):
                    post_num = len(self.visibility_filters[i])
                    self.visibility_filters[i] = self.visibility_filters[i][valid_mask[:post_num]]
            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)

            if new_keyframe and self.rerun_viewer:
                # current_i = copy.deepcopy(self.iter_shared[0])
                rgb_np = image.cpu().numpy().transpose(1, 2, 0)
                rgb_np = np.clip(rgb_np, 0., 1.0) * 255
                # rr.set_time_sequence("step", current_i)
                rr.set_time_seconds("log_time", time.time() - self.total_start_time_viewer)
                rr.log("rendered_rgb", rr.Image(rgb_np))

        self.training = False
        self.train_iter += 1

    def prepare_add_gaussian_mask(self, viewpoint_cam, tracking):
        silhouette, render_depth, render_depth_sq = self.check_silhouette(viewpoint_cam)
        render_pkg = render_3(viewpoint_cam, self.gaussians, self.pipe, self.background, training_stage=self.training_stage)
        # depth_image = render_pkg["render_depth"]
        render_image = render_pkg["render"]

        select_mask = (silhouette < 0.98)
        gt_depth = viewpoint_cam.original_depth_image.cuda()
        depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
        non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 10 * depth_error.median())  # 稠密mask，论文公式9
        gt_image = viewpoint_cam.original_image.cuda()
        rgb_error = torch.abs(gt_image - render_image)
        rgb_error = torch.sum(rgb_error, dim=0) * (gt_depth > 0)
        non_presence_rbg_mask = (rgb_error > 0.5)  # 稠密mask
        select_mask = select_mask | non_presence_depth_mask | non_presence_rbg_mask
        select_mask = select_mask.reshape(-1)   # .detach()

        points, colors, rots, scales, opacities, z_values, current_pose, tracking_mask, zero_filter, opacity_mask = self.shared_new_gaussians.get_values()

        high_freq_mask = self.high_freq_mask & opacity_mask
        low_freq_mask = self.low_freq_mask & (~opacity_mask)
        if tracking == True:
            select_mask = (high_freq_mask | low_freq_mask) & select_mask | self.tracking_down_mask# & uncertainty_mask
        else:
            select_mask = (high_freq_mask | low_freq_mask) & select_mask# & uncertainty_mask

        select_mask = select_mask[self.downsample_idxs_mapping][zero_filter].bool()

        points = points[select_mask, :]
        colors = colors[select_mask, :]
        rots = rots[select_mask, :]
        scales = scales[select_mask, :]
        opacities = opacities[select_mask, :]
        z_values = z_values[select_mask]
        if tracking == True:
            tracking_mask = tracking_mask[select_mask]
            tracking_filter = torch.where(tracking_mask)[0]
        else:
            tracking_filter = []

        return points, colors, rots, scales, opacities, z_values, current_pose, tracking_filter, render_pkg


def mse2psnr(x):
    return -10.*torch.log(x)/torch.log(torch.tensor(10.))
