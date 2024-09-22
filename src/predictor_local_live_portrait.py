# coding: utf-8

"""
Pipeline of LivePortrait (Human)
"""

import torch
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import numpy as np
import os
import os.path as osp

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.crop import prepare_paste_back, paste_back
from .utils.helper import dct2device
from .utils.rprint import rlog as log
# from .utils.viz import viz_lmk
from .live_portrait_wrapper import LivePortraitWrapper


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


class LivePortraitPipeline(object):

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
        self.cropper: Cropper = Cropper(crop_cfg=crop_cfg)
        self.start_frame = None
        self.start_frame_kp = None
        self.kp_driving_initial = None

    def prepare_for_saving(self, img):
        y = img[0].permute(1,2,0).cpu().numpy()
        y = np.clip(y , a_min=0, a_max=1)  # clip to 0~1
        y = y * 255
        return y.astype(np.uint8)

    def make_motion_template(self, I_lst, c_eyes_lst, c_lip_lst, **kwargs):
        n_frames = I_lst.shape[0]
        template_dct = {
            'n_frames': n_frames,
            'output_fps': kwargs.get('output_fps', 25),
            'motion': [],
            'c_eyes_lst': [],
            'c_lip_lst': [],
        }

        for i in range(n_frames):
            # collect s, R, δ and t for inference
            I_i = I_lst[i]
            # cv2.imwrite("frame_d.png", img=self.prepare_for_saving(I_i)[...,::-1])
            x_i_info = self.live_portrait_wrapper.get_kp_info(I_i)
            x_s = self.live_portrait_wrapper.transform_keypoint(x_i_info)
            R_i = get_rotation_matrix(x_i_info['pitch'], x_i_info['yaw'], x_i_info['roll'])

            item_dct = {
                'scale': x_i_info['scale'].cpu().numpy().astype(np.float32),
                'R': R_i.cpu().numpy().astype(np.float32),
                'exp': x_i_info['exp'].cpu().numpy().astype(np.float32),
                't': x_i_info['t'].cpu().numpy().astype(np.float32),
                'kp': x_i_info['kp'].cpu().numpy().astype(np.float32),
                'x_s': x_s.cpu().numpy().astype(np.float32),
            }

            template_dct['motion'].append(item_dct)

            c_eyes = c_eyes_lst[i].astype(np.float32)
            template_dct['c_eyes_lst'].append(c_eyes)

            c_lip = c_lip_lst[i].astype(np.float32)
            template_dct['c_lip_lst'].append(c_lip)

        return template_dct

    def reset_frames(self):
        self.kp_driving_initial = None
    
    def get_start_frame(self):
        return self.start_frame
    
    def get_frame_kp(self, image):
        print("get_frame_kp")
        print("#######NOT Implemented#######")
        pass

    @staticmethod
    def normalize_alignment_kp(kp):
        print("normalize")
        print("#######NOT Implemented#######")
        pass

    def get_start_frame_kp(self):
        print("get_start_frame_kp")
        print("#######NOT Implemented#######")
        pass

    def set_source_image(self, source_image):
        # for convenience
        inf_cfg = self.live_portrait_wrapper.inference_cfg
        device = self.live_portrait_wrapper.device
        crop_cfg = self.cropper.crop_cfg

        ######## load source input ########
        self.source_rgb_lst = [source_image]

        ######## prepare for pasteback ########

        self.flag_normalize_lip = inf_cfg.flag_normalize_lip  # not overwrite
        self.flag_source_video_eye_retargeting = inf_cfg.flag_source_video_eye_retargeting  # not overwrite
        self.lip_delta_before_animation, self.eye_delta_before_animation = None, None

        ######## process source info ########
        if inf_cfg.flag_do_crop:
            self.crop_info = self.cropper.crop_source_image(self.source_rgb_lst[0], crop_cfg)
            if self.crop_info is None:
                raise Exception("No face detected in the source image!")
            self.source_lmk = self.crop_info['lmk_crop']
            img_crop_256x256 = self.crop_info['img_crop_256x256']
        else:
            self.source_lmk = self.cropper.calc_lmk_from_cropped_image(self.source_rgb_lst[0])
            img_crop_256x256 = cv2.resize(self.source_rgb_lst[0], (256, 256))  # force to resize to 256x256
        I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
        self.x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
        self.x_c_s = self.x_s_info['kp']
        self.R_s = get_rotation_matrix(self.x_s_info['pitch'], self.x_s_info['yaw'], self.x_s_info['roll'])
        self.f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
        self.xs = self.live_portrait_wrapper.transform_keypoint(self.x_s_info)

        # let lip-open scalar to be 0 at first
        if self.flag_normalize_lip and inf_cfg.flag_relative_motion and self.source_lmk is not None:
            c_d_lip_before_animation = [0.]
            combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, self.source_lmk)
            if combined_lip_ratio_tensor_before_animation[0][0] >= inf_cfg.lip_normalize_threshold:
                self.lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(self.xs, combined_lip_ratio_tensor_before_animation)

        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            self.mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, self.crop_info['M_c2o'], dsize=(self.source_rgb_lst[0].shape[1], self.source_rgb_lst[0].shape[0]))

    def predict(self, driving_frame):
        # for convenience
        inf_cfg = self.live_portrait_wrapper.inference_cfg
        device = self.live_portrait_wrapper.device
        crop_cfg = self.cropper.crop_cfg


        ######## process driving info ########
        driving_rgb_crop_256x256_lst = None

        output_fps = 1
        driving_rgb_lst = [driving_frame]
        # else:
        #     raise Exception(f"{args.driving} is not a supported type!")
        ######## make motion template ########
        # log("Start making driving motion template...")
        driving_n_frames = len(driving_rgb_lst)

        n_frames = driving_n_frames

        driving_lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(driving_rgb_lst)
        driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]  # force to resize to 256x256
        #######################################

        c_d_eyes_lst, c_d_lip_lst = self.live_portrait_wrapper.calc_ratio(driving_lmk_crop_lst)
        # save the motion template
        # cv2.imwrite("frame_rgb_d.png", img=driving_rgb_crop_256x256_lst[0][...,::-1])
        I_d_lst = self.live_portrait_wrapper.prepare_videos(driving_rgb_crop_256x256_lst)
        driving_template_dct = self.make_motion_template(I_d_lst, c_d_eyes_lst, c_d_lip_lst, output_fps=output_fps)

        # wfp_template = remove_suffix(args.driving) + '.pkl'
        # dump(wfp_template, driving_template_dct)
        c_d_eyes_lst = c_d_eyes_lst*n_frames
        c_d_lip_lst = c_d_lip_lst*n_frames
        R_d_0, x_d_0_info = None, None

        ######## animate ########
    
        x_d_i_info = driving_template_dct['motion'][0]
        x_d_i_info = dct2device(x_d_i_info, device)
        R_d_i = x_d_i_info['R'] if 'R' in x_d_i_info.keys() else x_d_i_info['R_d']  # compatible with previous keys
        # print("##")
        # print(R_d_i)
        # print(self.R_s)
        # print(inf_cfg.driving_option)
        # print("##")

        R_d_0 = self.R_s
        x_d_0_info = x_d_i_info.copy()

        delta_new = self.x_s_info['exp'].clone()
        if inf_cfg.flag_relative_motion:
            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ self.R_s
            else:
                R_new = self.R_s
            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "exp":
                delta_new = self.x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(inf_cfg.lip_array).to(dtype=torch.float32, device=device))
            elif inf_cfg.animation_region == "lip":
                for lip_idx in [6, 12, 14, 17, 19, 20]:
                    delta_new[:, lip_idx, :] = (self.x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(inf_cfg.lip_array).to(dtype=torch.float32, device=device)))[:, lip_idx, :]
            elif inf_cfg.animation_region == "eyes":
                for eyes_idx in [11, 13, 15, 16, 18]:
                    delta_new[:, eyes_idx, :] = (self.x_s_info['exp'] + (x_d_i_info['exp'] - 0))[:, eyes_idx, :]
            if inf_cfg.animation_region == "all":
                scale_new = self.x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
            else:
                scale_new = self.x_s_info['scale']
            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                t_new = self.x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
            else:
                t_new = self.x_s_info['t']
        else:
            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                R_new = R_d_i
            else:
                R_new = self.R_s
            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "exp":
                for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
                    delta_new[:, idx, :] = x_d_i_info['exp'][:, idx, :]
                delta_new[:, 3:5, 1] = x_d_i_info['exp'][:, 3:5, 1]
                delta_new[:, 5, 2] = x_d_i_info['exp'][:, 5, 2]
                delta_new[:, 8, 2] = x_d_i_info['exp'][:, 8, 2]
                delta_new[:, 9, 1:] = x_d_i_info['exp'][:, 9, 1:]
            elif inf_cfg.animation_region == "lip":
                for lip_idx in [6, 12, 14, 17, 19, 20]:
                    delta_new[:, lip_idx, :] = x_d_i_info['exp'][:, lip_idx, :]
            elif inf_cfg.animation_region == "eyes":
                for eyes_idx in [11, 13, 15, 16, 18]:
                    delta_new[:, eyes_idx, :] = x_d_i_info['exp'][:, eyes_idx, :]
            scale_new = self.x_s_info['scale']
            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                t_new = x_d_i_info['t']
            else:
                t_new = self.x_s_info['t']

        t_new[..., 2].fill_(0)  # zero tz
        x_d_i_new = scale_new * (self.x_c_s @ R_new + delta_new) + t_new


        # Algorithm 1:
        if not inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
            # without stitching or retargeting
            if self.flag_normalize_lip and self.lip_delta_before_animation is not None:
                x_d_i_new += self.lip_delta_before_animation
            if self.flag_source_video_eye_retargeting and self.eye_delta_before_animation is not None:
                x_d_i_new += self.eye_delta_before_animation
            else:
                pass
        elif inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
            # with stitching and without retargeting
            if self.flag_normalize_lip and self.lip_delta_before_animation is not None:
                x_d_i_new = self.live_portrait_wrapper.stitching(self.xs, x_d_i_new) + self.lip_delta_before_animation
            else:
                x_d_i_new = self.live_portrait_wrapper.stitching(self.xs, x_d_i_new)
            if self.flag_source_video_eye_retargeting and self.eye_delta_before_animation is not None:
                x_d_i_new += self.eye_delta_before_animation
        else:
            eyes_delta, lip_delta = None, None
            if inf_cfg.flag_eye_retargeting and self.source_lmk is not None:
                c_d_eyes_i = c_d_eyes_lst[0]
                combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio(c_d_eyes_i, self.source_lmk)
                # ∆_eyes,i = R_eyes(self.xs; c_s,eyes, c_d,eyes,i)
                eyes_delta = self.live_portrait_wrapper.retarget_eye(self.xs, combined_eye_ratio_tensor)
            if inf_cfg.flag_lip_retargeting and self.source_lmk is not None:
                c_d_lip_i = c_d_lip_lst[0]
                combined_lip_ratio_tensor = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_i, self.source_lmk)
                # ∆_lip,i = R_lip(self.xs; c_s,lip, c_d,lip,i)
                lip_delta = self.live_portrait_wrapper.retarget_lip(self.xs, combined_lip_ratio_tensor)

            if inf_cfg.flag_relative_motion:  # use x_s
                x_d_i_new = self.xs + \
                    (eyes_delta if eyes_delta is not None else 0) + \
                    (lip_delta if lip_delta is not None else 0)
            else:  # use x_d,i
                x_d_i_new = x_d_i_new + \
                    (eyes_delta if eyes_delta is not None else 0) + \
                    (lip_delta if lip_delta is not None else 0)

            if inf_cfg.flag_stitching:
                x_d_i_new = self.live_portrait_wrapper.stitching(self.xs, x_d_i_new)

        x_d_i_new = self.xs + (x_d_i_new - self.xs) * inf_cfg.driving_multiplier
        out = self.live_portrait_wrapper.warp_decode(self.f_s, self.xs, x_d_i_new)
        I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]
        
        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            # TODO: the paste back procedure is slow, considering optimize it using multi-threading or GPU
            I_p_pstbk = paste_back(I_p_i, self.crop_info['M_c2o'], self.source_rgb_lst[0], self.mask_ori_float)
            return I_p_pstbk
        else:
            print("#NO!")
            return I_p_i
