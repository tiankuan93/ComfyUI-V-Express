import os
import cv2
import sys
import numpy as np
import time
import torch
import torchaudio.functional
import torchvision.io
from imageio_ffmpeg import get_ffmpeg_exe
from PIL import Image

from diffusers.utils.torch_utils import randn_tensor
from diffusers import AutoencoderKL
from insightface.app import FaceAnalysis
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import accelerate

import folder_paths
import folder_paths as comfy_paths
from comfy import model_management

ROOT_PATH = os.path.join(comfy_paths.get_folder_paths("custom_nodes")[0], "ComfyUI-V-Express")
sys.path.append(os.path.join(ROOT_PATH, 'src'))

from .src.pipelines import VExpressPipeline
from .src.pipelines.utils import draw_kps_image, save_video
from .src.pipelines.utils import retarget_kps
from .src.util import get_ffmpeg
from .src.inference import (
    get_scheduler,
    load_reference_net,
    load_denoising_unet,
    load_v_kps_guider,
    load_audio_projection,
)

INPUT_PATH = folder_paths.get_input_directory()
OUTPUT_PATH = folder_paths.get_output_directory()


INFERENCE_CONFIG_PATH = os.path.join(ROOT_PATH, "src/inference_v2.yaml")

load_device = model_management.get_torch_device()
offload_device = model_management.unet_offload_device()
DEVICE = load_device
WEIGHT_DTYPE = torch.float16
GPU_ID = 0

STANDARD_AUDIO_SAMPLING_RATE = 16000
NUM_PAD_AUDIO_FRAMES = 2


def get_all_model_path(vexpress_model_path):
    if not os.path.isabs(vexpress_model_path):
        vexpress_model_path = os.path.join(ROOT_PATH, vexpress_model_path)

    unet_config_path = os.path.join(vexpress_model_path, 'stable-diffusion-v1-5/unet/config.json')
    vae_path = os.path.join(vexpress_model_path, 'sd-vae-ft-mse')
    audio_encoder_path = os.path.join(vexpress_model_path, 'wav2vec2-base-960h')
    insightface_model_path = os.path.join(vexpress_model_path, 'insightface_models')

    denoising_unet_path = os.path.join(vexpress_model_path, 'v-express/denoising_unet.bin')
    reference_net_path = os.path.join(vexpress_model_path, 'v-express/reference_net.bin')
    v_kps_guider_path = os.path.join(vexpress_model_path, 'v-express/v_kps_guider.bin')
    audio_projection_path = os.path.join(vexpress_model_path, 'v-express/audio_projection.bin')
    motion_module_path = os.path.join(vexpress_model_path, 'v-express/motion_module.bin')

    if not os.path.isfile(denoising_unet_path):
        denoising_unet_path = os.path.join(vexpress_model_path, 'v-express/denoising_unet.pth')
    if not os.path.isfile(reference_net_path):
        reference_net_path = os.path.join(vexpress_model_path, 'v-express/reference_net.pth')
    if not os.path.isfile(v_kps_guider_path):
        v_kps_guider_path = os.path.join(vexpress_model_path, 'v-express/v_kps_guider.pth')
    if not os.path.isfile(audio_projection_path):
        audio_projection_path = os.path.join(vexpress_model_path, 'v-express/audio_projection.pth')
    if not os.path.isfile(motion_module_path):
        motion_module_path = os.path.join(vexpress_model_path, 'v-express/motion_module.pth')

    model_dict = {
        "unet_config_path": unet_config_path,
        "vae_path": vae_path,
        "audio_encoder_path": audio_encoder_path,
        "insightface_model_path": insightface_model_path,
        "denoising_unet_path": denoising_unet_path,
        "reference_net_path": reference_net_path,
        "v_kps_guider_path": v_kps_guider_path,
        "audio_projection_path": audio_projection_path,
        "motion_module_path": motion_module_path,
    }
    return model_dict


class VEINTConstant:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image_size": ("INT", {"default": 512, "min": 512, "max": 2048}),
        },
        }
    RETURN_TYPES = ("INT_INPUT",)
    RETURN_NAMES = ("image_size",)
    FUNCTION = "get_value"
    CATEGORY = "V-Express"

    def get_value(self, image_size):
        return (image_size,)


class VEStringConstant:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"default": './model_ckpts', "multiline": False}),
            }
        }
    RETURN_TYPES = ("STRING_INPUT",)
    FUNCTION = "passtring"
    CATEGORY = "V-Express"

    def passtring(self, string):
        return (string, )


class V_Express_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "v_express_pipeline": ("V_EXPRESS_PIPELINE",),
                "vexpress_model_path": ("STRING_INPUT", ),
                "audio_path": ("AUDIO_PATH",),
                "kps_path": ("VKPS_PATH",),
                "ref_image_path": ("IMAGE_PATH",),
                "output_path": ("STRING",{
                    "default": os.path.join(OUTPUT_PATH,f"{time.time()}_vexpress.mp4")
                }),
                "image_size": ("INT_INPUT",),
                "retarget_strategy": (
                    ["fix_face", "no_retarget", "offset_retarget", "naive_retarget"],
                    {"default": "fix_face"}
                ),
                "fps": ("FLOAT", {"default": 30.0, "min": 20.0, "max": 60.0}),
                "seed": ("INT",{
                    "default": 42
                }),
                "num_inference_steps": ("INT",{
                    "default": 20
                }),
                "guidance_scale": ("FLOAT",{
                    "default": 3.5
                }),
                "context_frames": ("INT",{
                    "default": 12
                }),
                "context_stride": ("INT",{
                    "default": 1
                }),
                "context_overlap": ("INT",{
                    "default": 4
                }),
                "reference_attention_weight": ("FLOAT",{
                    "default": 0.95
                }),
                "audio_attention_weight": ("FLOAT",{
                    "default": 3.
                }),
            }
        }

    RETURN_TYPES = (
        "STRING_INPUT",
    )
    RETURN_NAMES = (
        "output_path",
    )
    OUTPUT_NODE = True
    # OUTPUT_NODE = False
    CATEGORY = "V-Express"
    FUNCTION = "v_express"
    def v_express(
        self,
        v_express_pipeline,
        vexpress_model_path,
        audio_path,
        kps_path,
        ref_image_path,
        output_path,
        image_size,
        retarget_strategy,
        fps,
        seed,
        num_inference_steps,
        guidance_scale,
        context_frames,
        context_stride,
        context_overlap,
        reference_attention_weight,
        audio_attention_weight,
        save_gpu_memory=True,
        do_multi_devices_inference=False,
    ):
        start_time = time.time()

        accelerator = None

        reference_image_path = ref_image_path
        model_dict = get_all_model_path(vexpress_model_path)

        insightface_model_path = model_dict['insightface_model_path']

        app = FaceAnalysis(
            providers=['CUDAExecutionProvider' if DEVICE == 'cuda' else 'CPUExecutionProvider'],
            provider_options=[{'device_id': GPU_ID}] if DEVICE == 'cuda' else [],
            root=insightface_model_path,
        )
        app.prepare(ctx_id=0, det_size=(image_size, image_size))

        reference_image = Image.open(reference_image_path).convert('RGB')
        reference_image = reference_image.resize((image_size, image_size))

        reference_image_for_kps = cv2.imread(reference_image_path)
        reference_image_for_kps = cv2.resize(reference_image_for_kps, (image_size, image_size))
        reference_kps = app.get(reference_image_for_kps)[0].kps[:3]
        if save_gpu_memory:
            del app
        torch.cuda.empty_cache()

        _, audio_waveform, meta_info = torchvision.io.read_video(audio_path, pts_unit='sec')
        audio_sampling_rate = meta_info['audio_fps']
        print(f'Length of audio is {audio_waveform.shape[1]} with the sampling rate of {audio_sampling_rate}.')
        if audio_sampling_rate != STANDARD_AUDIO_SAMPLING_RATE:
            audio_waveform = torchaudio.functional.resample(
                audio_waveform,
                orig_freq=audio_sampling_rate,
                new_freq=STANDARD_AUDIO_SAMPLING_RATE,
            )
        audio_waveform = audio_waveform.mean(dim=0)

        duration = audio_waveform.shape[0] / STANDARD_AUDIO_SAMPLING_RATE
        init_video_length = int(duration * fps)
        num_contexts = np.around((init_video_length + context_overlap) / context_frames)
        video_length = int(num_contexts * context_frames - context_overlap)
        fps = video_length / duration
        print(f'The corresponding video length is {video_length}.')

        kps_sequence = None
        if kps_path != "":
            assert os.path.exists(kps_path), f'{kps_path} does not exist'
            kps_sequence = torch.tensor(torch.load(kps_path))  # [len, 3, 2]
            print(f'The original length of kps sequence is {kps_sequence.shape[0]}.')

            if kps_sequence.shape[0] > video_length:
                kps_sequence = kps_sequence[:video_length, :, :]

            kps_sequence = torch.nn.functional.interpolate(kps_sequence.permute(1, 2, 0), size=video_length, mode='linear')
            kps_sequence = kps_sequence.permute(2, 0, 1)
            print(f'The interpolated length of kps sequence is {kps_sequence.shape[0]}.')

        if retarget_strategy == 'fix_face':
            kps_sequence = torch.tensor([reference_kps] * video_length)
        elif retarget_strategy == 'no_retarget':
            kps_sequence = kps_sequence
        elif retarget_strategy == 'offset_retarget':
            kps_sequence = retarget_kps(reference_kps, kps_sequence, only_offset=True)
        elif retarget_strategy == 'naive_retarget':
            kps_sequence = retarget_kps(reference_kps, kps_sequence, only_offset=False)
        else:
            raise ValueError(f'The retarget strategy {retarget_strategy} is not supported.')

        kps_images = []
        for i in range(video_length):
            kps_image = draw_kps_image(image_size, image_size, kps_sequence[i])
            kps_images.append(Image.fromarray(kps_image))

        generator = torch.manual_seed(seed)
        video_tensor = v_express_pipeline(
            reference_image=reference_image,
            kps_images=kps_images,
            audio_waveform=audio_waveform,
            width=image_size,
            height=image_size,
            video_length=video_length,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            context_frames=context_frames,
            context_overlap=context_overlap,
            reference_attention_weight=reference_attention_weight,
            audio_attention_weight=audio_attention_weight,
            num_pad_audio_frames=NUM_PAD_AUDIO_FRAMES,
            generator=generator,
            do_multi_devices_inference=do_multi_devices_inference,
            save_gpu_memory=save_gpu_memory,
        )

        if accelerator is None or accelerator.is_main_process:
            save_video(video_tensor, audio_path, output_path, DEVICE, fps)
            consumed_time = time.time() - start_time
            generation_fps = video_tensor.shape[2] / consumed_time
            print(f'The generated video has been saved at {output_path}. '
                f'The generation time is {consumed_time:.1f} seconds. '
                f'The generation FPS is {generation_fps:.2f}.')

        return (output_path, )


class V_Express_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vexpress_model_path": ("STRING_INPUT", ),
            },
        }

    RETURN_TYPES = (
        "V_EXPRESS_PIPELINE",
    )
    RETURN_NAMES = (
        "v_express_pipeline",
    )

    CATEGORY = "V-Express"
    FUNCTION = "load_vexpress_pipeline"
    def load_vexpress_pipeline(self, vexpress_model_path):

        model_dict = get_all_model_path(vexpress_model_path)

        unet_config_path = model_dict['unet_config_path']
        reference_net_path = model_dict['reference_net_path']
        denoising_unet_path = model_dict['denoising_unet_path']
        v_kps_guider_path = model_dict['v_kps_guider_path']
        audio_projection_path = model_dict['audio_projection_path']
        motion_module_path = model_dict['motion_module_path']

        vae_path = model_dict['vae_path']
        audio_encoder_path = model_dict['audio_encoder_path']

        dtype = WEIGHT_DTYPE
        device = DEVICE
        inference_config_path = INFERENCE_CONFIG_PATH

        scheduler = get_scheduler(inference_config_path)
        reference_net = load_reference_net(unet_config_path, reference_net_path, dtype, device)
        denoising_unet = load_denoising_unet(
            inference_config_path, unet_config_path, denoising_unet_path, motion_module_path,
            dtype, device
        )
        v_kps_guider = load_v_kps_guider(v_kps_guider_path, dtype, device)
        audio_projection = load_audio_projection(
            audio_projection_path,
            dtype,
            device,
            inp_dim=denoising_unet.config.cross_attention_dim,
            mid_dim=denoising_unet.config.cross_attention_dim,
            out_dim=denoising_unet.config.cross_attention_dim,
            inp_seq_len=2 * (2 * NUM_PAD_AUDIO_FRAMES + 1),
            out_seq_len=2 * NUM_PAD_AUDIO_FRAMES + 1,
        )

        vae = AutoencoderKL.from_pretrained(vae_path).to(dtype=dtype, device=device)
        audio_encoder = Wav2Vec2Model.from_pretrained(audio_encoder_path).to(dtype=dtype, device=device)
        audio_processor = Wav2Vec2Processor.from_pretrained(audio_encoder_path)

        v_express_pipeline = VExpressPipeline(
            vae=vae,
            reference_net=reference_net,
            denoising_unet=denoising_unet,
            v_kps_guider=v_kps_guider,
            audio_processor=audio_processor,
            audio_encoder=audio_encoder,
            audio_projection=audio_projection,
            scheduler=scheduler,
        ).to(dtype=dtype, device=device)

        return (v_express_pipeline,)


class Load_Audio_Path:
    @classmethod
    def INPUT_TYPES(s):
        files = []
        for f in os.listdir(INPUT_PATH):
            if os.path.isfile(os.path.join(INPUT_PATH, f)) and f.split('.')[-1] in ["mp3"]: # only support mp3
                files.append(f)

        return {"required":{
            "audio_path": (files,),
        }}

    CATEGORY = "V-Express"

    RETURN_TYPES = ("AUDIO_PATH",)

    FUNCTION = "load_audio_path"
    def load_audio_path(self, audio_path):
        audio_path = os.path.join(INPUT_PATH, audio_path)
        return (audio_path,)


class Load_Audio_Path_From_Video:
    @classmethod
    def INPUT_TYPES(s):
        files = []
        for f in os.listdir(INPUT_PATH):
            if os.path.isfile(os.path.join(INPUT_PATH, f)) and f.split('.')[-1] in ["mp4", "webm","mkv","avi"]:
                files.append(f)

        return {"required":{
            "video_path": (files,),
        }}

    CATEGORY = "V-Express"

    RETURN_TYPES = ("AUDIO_PATH",)

    FUNCTION = "load_audio_path_from_video"
    def load_audio_path_from_video(self, video_path):
        video_path = os.path.join(INPUT_PATH, video_path)
        video_base_name = video_path[:video_path.rfind('.')]
        audio_name = f'{video_base_name}_audio.mp3'
        audio_path = os.path.join(INPUT_PATH, audio_name)
        os.system(f'{get_ffmpeg_exe()} -i "{video_path}" -y -vn "{audio_path}"')
        if not os.path.isfile(audio_path):
            raise ValueError(f'{audio_path} not exists! Please check if the video contains audio!')
        return (audio_path,)


class Load_Kps_Path:
    @classmethod
    def INPUT_TYPES(s):
        files = []
        for f in os.listdir(INPUT_PATH):
            if os.path.isfile(os.path.join(INPUT_PATH, f)) and f.split('.')[-1] in ["pth"]:
                files.append(f)

        return {"required":{
            "kps_path": (files,),
        }}

    CATEGORY = "V-Express"

    RETURN_TYPES = ("VKPS_PATH",)

    FUNCTION = "load_kps_path"
    def load_kps_path(self, kps_path):
        kps_path = os.path.join(INPUT_PATH, kps_path)
        return (kps_path,)


class Load_Kps_Path_From_Video:
    @classmethod
    def INPUT_TYPES(s):
        files = []
        for f in os.listdir(INPUT_PATH):
            if os.path.isfile(os.path.join(INPUT_PATH, f)) and f.split('.')[-1] in ["mp4", "webm","mkv","avi"]:
                files.append(f)

        return {"required":{
            "vexpress_model_path": ("STRING_INPUT", ),
            "video_path": (files,),
            "image_size": ("INT_INPUT",),
        }}

    CATEGORY = "V-Express"

    RETURN_TYPES = ("VKPS_PATH",)

    FUNCTION = "load_kps_path_from_video"
    def load_kps_path_from_video(self, vexpress_model_path, video_path, image_size):
        video_path = os.path.join(INPUT_PATH, video_path)
        video_base_name = video_path[:video_path.rfind('.')]
        kps_name = f'{video_base_name}_kps.pth'
        kps_path = os.path.join(INPUT_PATH, kps_name)

        model_dict = get_all_model_path(vexpress_model_path)
        insightface_model_path = model_dict['insightface_model_path']

        app = FaceAnalysis(
            providers=['CUDAExecutionProvider' if DEVICE == 'cuda' else 'CPUExecutionProvider'],
            provider_options=[{'device_id': GPU_ID}] if DEVICE == 'cuda' else [],
            root=insightface_model_path,
        )
        app.prepare(ctx_id=0, det_size=(image_size, image_size))

        kps_sequence = []
        video_capture = cv2.VideoCapture(video_path)
        frame_idx = 0
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            frame = cv2.resize(frame, (image_size, image_size))
            faces = app.get(frame)
            assert len(faces) == 1, f'There are {len(faces)} faces in the {frame_idx}-th frame. Only one face is supported.'

            kps = faces[0].kps[:3]
            kps_sequence.append(kps)
            frame_idx += 1
        torch.save(kps_sequence, kps_path)

        if not os.path.isfile(kps_path):
            raise ValueError(f'{kps_path} not exists! Please check the input!')
        return (kps_path,)


class Load_Image_Path:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = INPUT_PATH
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "V-Express"

    RETURN_TYPES = ("IMAGE_PATH",)
    FUNCTION = "load_image_path"
    def load_image_path(self, image):
        image_path = os.path.join(INPUT_PATH, image)
        return (image_path,)


class Load_Video_Path:
    @classmethod
    def INPUT_TYPES(s):
        files = []
        for f in os.listdir(INPUT_PATH):
            if os.path.isfile(os.path.join(INPUT_PATH, f)) and f.split('.')[-1] in ["mp4", "webm","mkv","avi"]:
                files.append(f)

        return {"required":{
            "video_path": (files,),
        }}

    CATEGORY = "V-Express"

    RETURN_TYPES = ("STRING_INPUT",)

    FUNCTION = "load_video_path"
    def load_video_path(self, video_path):
        video_path = os.path.join(INPUT_PATH, video_path)
        return (video_path,)


class VEPreview_Video:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "video":("STRING_INPUT",),
        }}

    CATEGORY = "V-Express"
    DESCRIPTION = "show result"

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "load_video"
    def load_video(self, video):
        video_name = os.path.basename(video)
        video_path_name = os.path.basename(os.path.dirname(video))
        return {"ui":{"video":[video_name, video_path_name]}}

    @classmethod
    def IS_CHANGED(s,):
       return ""


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "V_Express_Loader": V_Express_Loader,
    "V_Express_Sampler": V_Express_Sampler,
    "Load_Audio_Path": Load_Audio_Path,
    "Load_Audio_Path_From_Video": Load_Audio_Path_From_Video,
    "Load_Kps_Path": Load_Kps_Path,
    "Load_Kps_Path_From_Video": Load_Kps_Path_From_Video,
    "Load_Image_Path": Load_Image_Path,
    "Load_Video_Path": Load_Video_Path,
    "VEINTConstant": VEINTConstant,
    "VEStringConstant": VEStringConstant,
    "VEPreview_Video": VEPreview_Video,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "V_Express_Loader": "V-Express Loader",
    "V_Express_Sampler": "V-Express Sampler",
    "Load_Audio_Path": "Load Audio Path",
    "Load_Audio_Path_From_Video": "Load Audio Path From Video",
    "Load_Kps_Path": "Load V-Kps Path",
    "Load_Kps_Path_From_Video": "Load V-Kps Path From Video",
    "Load_Image_Path": "Load Reference Image Path",
    "Load_Video_Path": "Load Video Path",
    "VEINTConstant": "Set Image Size",
    "VEStringConstant": "Set V-Express Model Path",
    "VEPreview_Video": "Preview Output Video",
}