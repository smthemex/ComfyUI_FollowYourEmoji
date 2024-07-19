import os

import sys
import imageio
import argparse
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
import cv2
import random
import decord
import matplotlib.pyplot as plt
from io import BytesIO
from IPython.display import Video
from IPython.display import display, Image as IPyImage

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torchvision.transforms as T
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available

from transformers import CLIPVisionModelWithProjection

from .models.guider import Guider
from .models.referencenet import ReferenceNet2DConditionModel
from .models.unet import UNet3DConditionModel
from .models.video_pipeline import VideoPipeline
from .dataset.val_dataset import ValDataset, val_collate_fn
from .media_pipe.mp_utils  import LMKExtractor
from .media_pipe.draw_util import FaceMeshVisualizer
from .media_pipe.pose_util import project_points_with_trans, matrix_to_euler_and_translation, euler_and_translation_to_matrix
from tqdm import tqdm


import folder_paths
emoji_current_path = os.path.dirname(os.path.abspath(__file__))
node_path_dir = os.path.dirname(emoji_current_path)
comfy_file_path = os.path.dirname(node_path_dir)
MAX_SEED = np.iinfo(np.int32).max
weigths_current_path = os.path.join(folder_paths.models_dir, "follow_emoji")

if not os.path.exists(weigths_current_path):
    os.makedirs(weigths_current_path)
    
def get_video_img(tensor):
    outputs = []
    for x in tensor:
        x = tensor_to_pil(x)
        outputs.append(x)
    return outputs
def find_directories(base_path):
    directories = []
    for root, dirs, files in os.walk(base_path):
        for name in dirs:
            directories.append(name)
    return directories

npy_path=os.path.join(folder_paths.input_directory,"emoji")
if not os.path.exists(npy_path):
    os.makedirs(npy_path)
    
npy_list = find_directories(npy_path)
if npy_list:
    npy_file_list=["none"]+npy_list
else:
    npy_file_list=["none",]


config_path = os.path.join(emoji_current_path, "configs/infer.yaml")
config = OmegaConf.load(config_path)

def pil2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def narry_list(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        modified_value = pil2narry(value)
        list_in[i] = modified_value
    return list_in

def tensor_to_pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image


# def show_img(img, title=''):
#     plt.figure(figsize=(10, 10))
#     plt.title(title)
#     plt.imshow(img)
#     plt.show()


def read_video(video_path):
    vr = decord.VideoReader(video_path)
    frames = vr.get_batch(list(range(len(vr))))
    return frames

# def display_gif(image_array, duration=100):
#     # 将NumPy数组转换为PIL图像列表
#     images = [Image.fromarray(frame) for frame in image_array]
#
#     # 将图像保存为GIF并将其读入内存缓冲区
#     buffer = BytesIO()
#     images[0].save(buffer, format='GIF', save_all=True, append_images=images[1:], duration=duration, loop=0)
#
#     # 在Jupyter Notebook中显示GIF
#     buffer.seek(0)
#     display(IPyImage(data=buffer.getvalue()))
    
# def make_temple_image(lmk_extractor,vis,image,dir_random): #单图测试，不需要
#     image = tensor_to_pil(image)
#     ref_image_pil = image.convert("RGB")
#     ref_image_np = cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR)
#     face_result = lmk_extractor(ref_image_np)
#     assert face_result is not None, "Can not detect a face in the reference image."
#     face_result['width'] = ref_image_np.shape[1]
#     face_result['height'] = ref_image_np.shape[0]
#
#     npy_name = "npy_" + ''.join(random.choice("0123456789") for _ in range(5))
#     input_path_img=os.path.join(npy_path,dir_random,"input")
#     if not os.path.exists(input_path_img):
#         os.makedirs(input_path_img)
#     save_path =os.path.join(npy_path,dir_random,"lmk",f"{npy_name}_mppose.npy")
#
#     np.save(save_path, face_result)
#     print(f"saving {npy_name}_mppose.npy" in {npy_path})
#
#     lmks = face_result['lmks'].astype(np.float32)
#     ref_pose = vis.draw_landmarks((ref_image_np.shape[1], ref_image_np.shape[0]), lmks, normed=True)
#     return input_path_img,save_path
    
def make_temple_video(lmk_extractor,vis,video_path,dir_random):
    
    frames = imageio.get_reader(video_path)
    face_results = []
    motions = []
    for frame in tqdm(frames):
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        
        face_result = lmk_extractor(frame_bgr)
        assert face_result is not None, "Can not detect a face in the reference image."
        face_result['width'] = frame_bgr.shape[1]
        face_result['height'] = frame_bgr.shape[0]
        
        face_results.append(face_result)
        lmks = face_result['lmks'].astype(np.float32)
        motion = vis.draw_landmarks((frame_bgr.shape[1], frame_bgr.shape[0]), lmks, normed=True)
        motions.append(motion)
    print(len(motions))
    save_video_path = os.path.splitext(video_path)[0]
    save_path= os.path.normpath(f"{save_video_path}_mppose.gif")
    
    imageio.mimsave(save_path, motions, 'GIF', duration=0.2, loop=0)
    save_dir = os.path.join(npy_path, dir_random)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    npy_save_path = os.path.join(save_dir, f"{dir_random}_mppose.npy")
    np.save(npy_save_path, face_results)
    print(f'Read {npy_save_path},done')
    return save_dir

def load_model_state_dict(model, model_ckpt_path, name):
    ckpt = torch.load(model_ckpt_path, map_location="cpu")
    model_state_dict = model.state_dict()
    model_new_sd = {}
    count = 0
    for k, v in ckpt.items():
        if k in model_state_dict:
            count += 1
            model_new_sd[k] = v
    miss, _ = model.load_state_dict(model_new_sd, strict=False)
    print(f'load {name} from {model_ckpt_path}\n - load params: {count}\n - miss params: {miss}')


@torch.no_grad()
def visualize(dataloader, pipeline, generator, W, H, video_length, num_inference_steps, guidance_scale, output_dir, limit=1,fps=8,save_video=False):

    for i, batch in enumerate(dataloader):
        ref_frame=batch['ref_frame'][0]
        clip_image = batch['clip_image'][0]
        motions=batch['motions'][0]
        file_name = batch['file_name'][0]
        if motions is None:
            continue
        if 'lmk_name' in batch:
            lmk_name = batch['lmk_name'][0].split('.')[0]
        else:
            lmk_name = 'lmk'
        print(file_name, lmk_name)
        # tensor to pil image
        ref_frame = torch.clamp((ref_frame + 1.0) / 2.0, min=0, max=1)
        ref_frame = ref_frame.permute((1, 2, 3, 0)).squeeze()
        ref_frame = (ref_frame * 255).cpu().numpy().astype(np.uint8)
        ref_image = Image.fromarray(ref_frame)
        # tensor to pil image
        motions = motions.permute((1, 2, 3, 0))
        motions = (motions * 255).cpu().numpy().astype(np.uint8)
        lmk_images = []
        for motion in motions:
            lmk_images.append(Image.fromarray(motion))

        preds = pipeline(ref_image=ref_image,
                        lmk_images=lmk_images,
                        width=W,
                        height=H,
                        video_length=video_length,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        clip_image=clip_image,
                        ).videos
        
        preds_tensor = preds.permute((0,2,3,4,1)).squeeze(0)
        preds=get_video_img(preds_tensor)
       
        #print(type(preds))
        if save_video:
            preds = (preds_tensor * 255).cpu().numpy().astype(np.uint8)
            mp4_path = os.path.join(output_dir, lmk_name+'_'+file_name.split('.')[0]+'_oo.mp4')
            mp4_writer = imageio.get_writer(mp4_path, fps=fps)
            for pred in preds:
                mp4_writer.append_data(pred)
            mp4_writer.close()
            
        #对比，不需要
        # mp4_path = os.path.join(output_dir, lmk_name+'_'+file_name.split('.')[0]+'_all.mp4')
        # mp4_writer = imageio.get_writer(mp4_path, fps=8)
        # if 'frames' in batch:
        #     frames = batch['frames'][0]
        #     frames = torch.clamp((frames + 1.0) / 2.0, min=0, max=1)
        #     frames = frames.permute((1, 2, 3, 0))
        #     frames = (frames * 255).cpu().numpy().astype(np.uint8)
        #     for frame, motion, pred in zip(frames, motions, preds):
        #         out = np.concatenate((frame, motion, ref_frame, pred), axis=1)
        #         mp4_writer.append_data(out)
        # else:
        #     for motion, pred in zip(motions, preds):
        #         out = np.concatenate((motion, ref_frame, pred), axis=1)
        #         mp4_writer.append_data(out)
        # mp4_writer.close()

        if i >= limit:
            break
        return preds

def get_instance_path(path):
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = instance_path.replace('\\', "/")
    return instance_path


def download_weights(file_dir,repo_id,subfolder="",pt_name=""):
    if subfolder:
        file_path = os.path.join(file_dir,subfolder, pt_name)
        sub_dir=os.path.join(file_dir,subfolder)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        if not os.path.exists(file_path):
            pt_path = hf_hub_download(
                repo_id=repo_id,
                filename=pt_name,
                local_dir = sub_dir,
            )
        else:
            pt_path=get_instance_path(file_path)
        return pt_path
    else:
        file_path = os.path.join(file_dir, pt_name)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        if not os.path.exists(file_path):
            pt_path = hf_hub_download(
                repo_id=repo_id,
                filename=pt_name,
                local_dir=file_dir,
            )
        else:
            pt_path=get_instance_path(file_path)
        return pt_path
    
class FollowYouEmoji_LoadModel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae":("STRING", {"default": "stabilityai/sd-vae-ft-mse"}),
                "weight_dtype": (["fp16", "fp32"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "main"
    CATEGORY = "FollowYouEmoji"
    
    def main(self,vae,weight_dtype):
        
        if weight_dtype == "fp16":
            weight_dtype = torch.float16
        else:
            weight_dtype = torch.float32
 
        # normal vae
        vae = AutoencoderKL.from_pretrained(vae).to(dtype=weight_dtype, device="cuda")
        
        #pre base model
        download_weights(weigths_current_path,"lambdalabs/sd-image-variations-diffusers","image_encoder","pytorch_model.bin")
        download_weights(weigths_current_path, "lambdalabs/sd-image-variations-diffusers", "image_encoder",
                         "config.json")
        image_encode_path=os.path.join(weigths_current_path,"image_encoder")
        
        motion_module_path = download_weights(weigths_current_path, "guoyww/animatediff",pt_name="mm_sd_v15_v2.ckpt")
        
        download_weights(weigths_current_path,"lambdalabs/sd-image-variations-diffusers","unet","diffusion_pytorch_model.bin")
        download_weights(weigths_current_path, "lambdalabs/sd-image-variations-diffusers", "unet",
                         "config.json")
        
        #unet_model_path = os.path.join(weigths_current_path, "unet")
        reference_pt_path=download_weights(weigths_current_path,"YueMafighting/FollowYourEmoji","ckpts","referencenet.pth")
        unet_ckpt_path=download_weights(weigths_current_path,"YueMafighting/FollowYourEmoji","ckpts","unet.pth")
        pose_guider_path=download_weights(weigths_current_path,"YueMafighting/FollowYourEmoji","ckpts","lmk_guider.pth")
        
        # init model
        # print('init model')
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encode_path).to(dtype=weight_dtype,
                                                                                                    device="cuda")
        referencenet = ReferenceNet2DConditionModel.from_pretrained_2d(weigths_current_path, subfolder="unet",
                                                                       referencenet_additional_kwargs=config.model.referencenet_additional_kwargs).to(
            device="cuda")
        
        unet = UNet3DConditionModel.from_pretrained_2d(weigths_current_path,
                                                       motion_module_path= motion_module_path, subfolder="unet",
                                                       unet_additional_kwargs=config.model.unet_additional_kwargs).to(
            device="cuda")
        
        lmk_guider = Guider(conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)).to(device="cuda")
        
        # load model
        #print('load model')
        load_model_state_dict(referencenet, reference_pt_path, 'referencenet')
        load_model_state_dict(unet, unet_ckpt_path, 'unet')
        load_model_state_dict(lmk_guider, pose_guider_path, 'lmk_guider')
        
        if config.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                referencenet.enable_xformers_memory_efficient_attention()
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        
        unet.set_reentrant(use_reentrant=False)
        referencenet.set_reentrant(use_reentrant=False)
        
        vae.eval()
        image_encoder.eval()
        unet.eval()
        referencenet.eval()
        lmk_guider.eval()
        
        # noise scheduler
        #print('init noise scheduler')
        sched_kwargs = OmegaConf.to_container(config.scheduler)
        if config.enable_zero_snr:
            sched_kwargs.update(rescale_betas_zero_snr=True,
                                timestep_spacing="trailing",
                                prediction_type="v_prediction")
        noise_scheduler = DDIMScheduler(**sched_kwargs)
        
        # pipeline
        pipelines = VideoPipeline(vae=vae,
                                 image_encoder=image_encoder,
                                 referencenet=referencenet,
                                 unet=unet,
                                 lmk_guider=lmk_guider,
                                 scheduler=noise_scheduler).to(vae.device, dtype=weight_dtype)
        return (pipelines,)


class Emoji_Make_Temple:
    @classmethod
    def INPUT_TYPES(s):
        input_path = folder_paths.get_input_directory()
        video_files = [f for f in os.listdir(input_path) if
                       os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ['webm', 'mp4', 'mkv',
                                                                                            'gif']]
        return {
            "required": {
                "npy_file": (npy_file_list,),
            },
            "optional": {
                 "video_files": (["none"] + video_files,),}
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lmk",)
    FUNCTION = "temple_main"
    CATEGORY = "FollowYouEmoji"
    
    def temple_main(self,npy_file,**kwargs):
        dir_random = "emoji_" + ''.join(random.choice("0123456789") for _ in range(6))
        video_files=kwargs.get("video_files")
        
        video_path=os.path.join(folder_paths.input_directory,video_files)
        
        lmk_extractor = LMKExtractor()
        vis = FaceMeshVisualizer(forehead_edge=False)
        if video_files!="none":
            if npy_file!="none":
                save_path = get_instance_path(os.path.join(npy_path, npy_file))
            else:
                save_path=make_temple_video(lmk_extractor, vis, video_path,dir_random)
        else:
            if npy_file!="none":
                save_path = get_instance_path(os.path.join(npy_path, npy_file))
            else:
                raise "need a video or npy_dir"
        lmk=";".join([save_path,dir_random])
        return (lmk,)
    

class FollowYouEmoji_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipelines": ("MODEL",),
                "image": ("IMAGE",),
                "lmk":("STRING", {"forceInput": True, "default": ""}),
                "seed": ("INT", {"default": 42, "min": 0, "max": MAX_SEED}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 30.0, "step": 0.1, "round": 0.01}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "fps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "length": ("INT", {"default": 16, "min": 4, "max": 160, "step": 1, "display": "number"}),
                "width": ("INT", {"default": 512, "min": 256, "max": 768, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 512, "min": 256, "max": 768, "step": 64, "display": "number"}),
                "save_video": ("BOOLEAN", {"default": False},),
            }
        }
    
    RETURN_TYPES = ("IMAGE","FLOAT")
    RETURN_NAMES = ("image","frame_rate")
    FUNCTION = "emoji_main"
    CATEGORY = "FollowYouEmoji"
    
    def emoji_main(self,pipelines,image,lmk,seed,cfg,steps,fps,length,height,width,save_video):
        
        # dataset creation
        #单cuda，不需要初始化，
        # local_rank = int(os.environ['LOCAL_RANK'])
        # dist.init_process_group(backend='nccl',rank=local_rank, world_size=1)
        # local_rank = int(os.environ['LOCAL_RANK'])
        # torch.cuda.set_device(local_rank)
        #
        # output_path = folder_paths.output_directory
        # if dist.get_rank() == 0:
        #     if not os.path.exists(output_path):
        #         os.makedirs(output_path, exist_ok=True)
        
        #print(type(lmk),lmk)
        lmk,dir_random = lmk.split(";")
        img_file_name=dir_random+".png"
        image=tensor_to_pil(image)
        lmk=os.path.normpath(lmk)
        input=os.path.join(lmk,img_file_name)
        image.save(input)
        output_path=folder_paths.output_directory
        
        val_dataset = ValDataset(
            input_path=lmk,
            lmk_path=lmk,
            resolution_h=height,
            resolution_w=width
        )
        print(len(val_dataset))
        #sampler = DistributedSampler(val_dataset, shuffle=False) #如上
        # DataLoaders creation:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=val_collate_fn,
        )
        #如上
        # val_dataloader = DataLoader(
        #     val_dataset,
        #     batch_size=1,
        #     num_workers=0,
        #     sampler=sampler,
        #     collate_fn=val_collate_fn,
        # )
        
        vae=pipelines.vae
        generator = torch.Generator(device=vae.device)
        generator.manual_seed(seed)
        
        # run visualize
        #print('run visualize')
        with torch.no_grad():
            preds=visualize(val_dataloader,
                      pipelines,
                      generator,
                      W=width,
                      H=height,
                      video_length=length,
                      num_inference_steps=steps,
                      guidance_scale=cfg,
                      output_dir=output_path,
                      limit=100000000,
                      fps=fps,
                      save_video=save_video)
        gen = narry_list(preds)  # pil列表排序
        images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
        return (images,fps)

    
NODE_CLASS_MAPPINGS = {
    "FollowYouEmoji_LoadModel":FollowYouEmoji_LoadModel,
    "Emoji_Make_Temple":Emoji_Make_Temple,
    "FollowYouEmoji_Sampler": FollowYouEmoji_Sampler
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FollowYouEmoji_LoadModel":"FollowYouEmoji_LoadModel",
    "Emoji_Make_Temple":"Emoji_Make_Temple",
    "FollowYouEmoji_Sampler": "FollowYouEmoji_Sampler"
}
