# ComfyUI_FollowYourEmoji

You can using FollowYourEmoji in comfyui   
FollowYourEmoji：Fine-Controllable and Expressive Freestyle Portrait Animation  
--

FollowYourEmoji  From: [FollowYourEmoji](https://github.com/mayuelala/FollowYourEmoji/tree/main)
--

My ComfyUI node list：
-----
1、ParlerTTS node:[ComfyUI_ParlerTTS](https://github.com/smthemex/ComfyUI_ParlerTTS)     
2、Llama3_8B node:[ComfyUI_Llama3_8B](https://github.com/smthemex/ComfyUI_Llama3_8B)      
3、HiDiffusion node：[ComfyUI_HiDiffusion_Pro](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro)   
4、ID_Animator node： [ComfyUI_ID_Animator](https://github.com/smthemex/ComfyUI_ID_Animator)       
5、StoryDiffusion node：[ComfyUI_StoryDiffusion](https://github.com/smthemex/ComfyUI_StoryDiffusion)  
6、Pops node：[ComfyUI_Pops](https://github.com/smthemex/ComfyUI_Pops)   
7、stable-audio-open-1.0 node ：[ComfyUI_StableAudio_Open](https://github.com/smthemex/ComfyUI_StableAudio_Open)        
8、GLM4 node：[ComfyUI_ChatGLM_API](https://github.com/smthemex/ComfyUI_ChatGLM_API)   
9、CustomNet node：[ComfyUI_CustomNet](https://github.com/smthemex/ComfyUI_CustomNet)           
10、Pipeline_Tool node :[ComfyUI_Pipeline_Tool](https://github.com/smthemex/ComfyUI_Pipeline_Tool)    
11、Pic2Story node :[ComfyUI_Pic2Story](https://github.com/smthemex/ComfyUI_Pic2Story)   
12、PBR_Maker node:[ComfyUI_PBR_Maker](https://github.com/smthemex/ComfyUI_PBR_Maker)      
13、ComfyUI_Streamv2v_Plus node:[ComfyUI_Streamv2v_Plus](https://github.com/smthemex/ComfyUI_Streamv2v_Plus)   
14、ComfyUI_MS_Diffusion node:[ComfyUI_MS_Diffusion](https://github.com/smthemex/ComfyUI_MS_Diffusion)   
15、ComfyUI_AnyDoor node: [ComfyUI_AnyDoor](https://github.com/smthemex/ComfyUI_AnyDoor)  
16、ComfyUI_Stable_Makeup node: [ComfyUI_Stable_Makeup](https://github.com/smthemex/ComfyUI_Stable_Makeup)  
17、ComfyUI_EchoMimic node:  [ComfyUI_EchoMimic](https://github.com/smthemex/ComfyUI_EchoMimic)   
18、ComfyUI_FollowYourEmoji node: [ComfyUI_FollowYourEmoji](https://github.com/smthemex/ComfyUI_FollowYourEmoji)   

Tips：
---
---选择你要参考的表情包视频，选择参考图片，运行；   
---基于参考视频生成的npy文件，在input/emoji目录下，再次使用，可以在“npy_file”菜单中选择（如果不显示，重启comfyUI）   
---length 生成视频时长，save_video 是否保存文件；  
---“npy_file” 和“video_file”选项均不是none时，“npy_file”优先。   
---Select the emoji video you want to reference, choose the reference image, and run it;   
---The npy file generated based on the reference video can be used again in the input/emoji directory by selecting it from the "npy_file" menu (if not displayed, restart comfyUI)；   
---Generate video duration using 'length', whether to save the file with 'save_video'；  
---When both the 'npy_file' and 'videoFILE' options are not 'none', 'npy_file' takes priority.   



1.Installation
-----
  In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_FollowYourEmoji.git
```  
  
2.requirements  
----
```
pip install -r requirements.txt

```
  
缺啥装啥。。。  
If the module is missing, , pip install  missing module.       

3 Need  model 
----
如果能直连抱脸，或者直连镜像站,点击就会自动下载所需模型,不需要单独下载.   

unet与ComfyUI_EchoMimic的一样  unet is the same as ComfyUI-EchoMimic    
unet [link](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)  

image_encoder [link](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)  

vae  stabilityai/sd-vae-ft-mse [link](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main)

mm_sd_v15_v2.ckpt    [link](https://huggingface.co/guoyww/animatediff/tree/main)

main model  [link](https://huggingface.co/YueMafighting/FollowYourEmoji/tree/main/ckpts)   

所用模型存放地址（Storage address of the model used）（all 14.1G）:  
```
├── ComfyUI/models/  
|     ├──follow_emoji
|         ├── unet
|             ├── diffusion_pytorch_model.bin
|             ├── config.json
|         ├── image_encoder
|             ├── pytorch_model.bin
|             ├── config.json
|         ├── ckpts
|             ├── lmk_guider.pth
|             ├── referencenet.pth
|             ├── unet.pth
|         ├── mm_sd_v15_v2.ckpt
```




Example
-----
make emoji from video  从参考视频生成emoji       
![](https://github.com/smthemex/ComfyUI_FollowYourEmoji/blob/main/example/Animate.gif)



6 Citation   
If you find Follow-Your-Emoji useful for your research, welcome to 🌟 FollowYourEmoji repo and cite our work using the following BibTeX:
------
``` python  
@article{ma2024follow,
  title={Follow-Your-Emoji: Fine-Controllable and Expressive Freestyle Portrait Animation},
  author={Ma, Yue and Liu, Hongyu and Wang, Hongfa and Pan, Heng and He, Yingqing and Yuan, Junkun and Zeng, Ailing and Cai, Chengfei and Shum, Heung-Yeung and Liu, Wei and others},
  journal={arXiv preprint arXiv:2406.01900},
  year={2024}
}
```


