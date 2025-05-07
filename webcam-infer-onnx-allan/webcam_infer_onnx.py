import random
from threading import Thread
import numpy as np
import builtins
import colorsys
import cv2
import time
import math
from queue import Queue, Empty
from datetime import datetime
import onnx
#import onnxoptimizer
import torch
from onnx import shape_inference
import onnxruntime as ort
import multiprocessing as mp
import os


os.environ['TRIDENT_BACKEND'] = 'pytorch'
from math import cos, sin
import trident
from trident import *
from trident.data.image_common import image_backend_adaption, reverse_image_backend_adaption
# from render_utilis.uv import uv_tex
# from render_utilis.render_ctypes import render
# from render_utilis.serialization import ser_to_ply, ser_to_obj
# from render_utilis.render_ctypes import render
#

hair_colors=OrderedDict()
hair_colors_imgs=glob.glob('./hair/*.jpg')
for colors_img in hair_colors_imgs:
    folder,filename,ext=split_path(colors_img)
    img_hsv= cv2.cvtColor(image2array(colors_img).astype(np.float32), cv2.COLOR_RGB2HSV)
    hsv_mean=img_hsv.copy().reshape((-1,3)).mean(0)
    hsv_std=np.clip(img_hsv.copy().reshape((-1,3)).std(0),0,20)

    hair_colors[filename]=[hsv_mean,hsv_std]




from skimage.filters import gaussian
def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)

def generate_palette(num_classes: int, format: str = 'rgb'):
    """Generate palette used in visualization.

    Args:
        num_classes (int): numbers of colors need in palette?
        format (string):  'rgb' by pixel tuple or  'hex'  by hex formatted color.

    Returns:
        colors in rgb or hex format
    Examples:
        >>> generate_palette(10)
        [(128, 64, 64), (128, 102, 64), (115, 128, 64), (77, 128, 64), (64, 128, 89), (64, 128, 128), (64, 89, 128), (76, 64, 128), (115, 64, 128), (128, 64, 102)]
        >>> generate_palette(24,'hex')
        ['#804040', '#805040', '#806040', '#807040', '#808040', '#708040', '#608040', '#508040', '#408040', '#408050', '#408060', '#408070', '#408080', '#407080', '#406080', '#405080', '#404080', '#504080', '#604080', '#704080', '#804080', '#804070', '#804060', '#804050']

    """

    def hex_format(rgb_tuple):
        return '#{:02X}{:02X}{:02X}'.format(rgb_tuple[0], rgb_tuple[1], rgb_tuple[2])

    hsv_tuples = [(x / num_classes, 1.0, 1.0) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (
        int(builtins.round(x[0] * 255.)), int(builtins.round(x[1] * 255.)), int(builtins.round(x[2] * 255.))), colors))
    if format == 'rgb':
        return colors
    elif format == 'hex':
        return [hex_format(color) for color in colors]


def crop_resize(frame, offsetx=0, offsety=210, cropx=540, cropy=540, tox=192, toy=192):
    return cv2.resize(frame.copy()[offsety:offsety + cropy, offsetx:offsetx + cropx, :], (tox, toy), cv2.INTER_AREA)


FACE_POINTS = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33])-1 #臉的輪廓
UPPER_LIP = (np.array([85, 86, 87, 88, 89, 90, 91, 100, 99, 98, 85]) - 1).tolist()  # 嘴
LOWER_LIP = (np.array([85, 97, 104, 103, 102, 101, 91, 92, 93, 94, 95, 96, 85]) - 1).tolist()  # 嘴
RIGHT_BROW_POINTS = (np.array([43, 44, 45, 46, 47, 48, 49, 50, 51, 43]) - 1).tolist()  # 右眉毛
LEFT_BROW_POINTS = (np.array([34, 35, 36, 37, 38, 39, 40, 41, 42, 34]) - 1).tolist()  # 左眉毛
RIGHT_EYE_POINTS = np.array([76, 77, 78, 79, 80, 81, 82, 83, 76]).astype(np.int64) - 1  # 右眼
LEFT_EYE_POINTS = np.array([67, 68, 69, 70, 71, 72, 73, 74, 67]).astype(np.int64) - 1  # 左眼
NOSE_POINTS = (np.array([52, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 52]) - 1).tolist()  # 鼻子

def get_image_mask(landmarks, image, points=[], color=(255, 255, 255)):

    for points in points:
        try:
            image = cv2.fillPoly(image, pts=np.array([landmarks[points, :]]).astype(np.int32),color=color)
        except Exception as e:
            print(e)
    return image



def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def chage_color(hsv_image,mask,img_hsv_str,hair_hsv_str):
    mask = mask / mask.max()
    hair_hsv_str=np.expand_dims(np.expand_dims(hair_hsv_str,0),0)
    img_hsv_str = np.expand_dims(np.expand_dims(img_hsv_str, 0), 0)
    hair_hsv_minmax=(hair_hsv_str[:,:,:,2:].copy()-hair_hsv_str[:,:,:,0:1].copy())/hair_hsv_str[:,:,:,1:2].copy()
    img_hsv_minmax = (img_hsv_str[:,:,:, 2:].copy() - img_hsv_str[:,:,:, 0:1]).copy() / img_hsv_str[:,:,:, 1:2].copy()

    norm_hsv_image=(hsv_image.copy()-img_hsv_str[:,:,:,0])/img_hsv_str[:,:,:,1]

    norm_hsv_image=(norm_hsv_image-img_hsv_minmax[:,:,:,0])/(img_hsv_minmax[:,:,:,1]-img_hsv_minmax[:,:,:,0])

    norm_hsv_image=norm_hsv_image*(hair_hsv_minmax[:,:,:,1]-hair_hsv_minmax[:,:,:,0])+hair_hsv_minmax[:,:,:,0]
    norm_hsv_image =(norm_hsv_image*hair_hsv_str[:,:,:,1]+hair_hsv_str[:,:,:,0]).astype(np.float32)

    # lv=hair_hsv_str[0, 0, 2, 2].item()
    # uv=hair_hsv_str[0, 0, 2, 3].item()
    # norm_hsv_image[:, :, 2] = np.clip(norm_hsv_image[:, :, 2] ,lv,uv)



    norm_hsv_image[:,:,0]=hair_hsv_str[:,:,0,0]
    norm_hsv_image[:, :, 1] = np.clip(norm_hsv_image[:, :, 1], 0, 1)
    norm_hsv_image[:, :, 2] = np.clip(norm_hsv_image[:, :, 2], 0, 255)
    changed_base=norm_hsv_image[mask==1,:]
    changed = cv2.cvtColor(norm_hsv_image.copy(), cv2.COLOR_HSV2RGB).astype(np.float32)

    changed_hsv_str = np.array(list(zip(changed_base.reshape((-1, 3)).mean(0), changed_base.reshape((-1, 3)).std(0), changed_base.reshape((-1, 3)).min(0),changed_base.reshape((-1, 3)).max(0))))

    return changed,changed_hsv_str
def stat2str(stat):
    return 'mean:{0:.2f}  std:{1:.2f}   min:{2:.2f}  max:{3:.2f}'.format(stat[0],stat[1],stat[2],stat[3])

def hsv2str(hsv_array):
    return "h:{0},\ns:{1},\nv:{2}".format(stat2str(hsv_array[0,:]),stat2str(hsv_array[1,:]),stat2str(hsv_array[2,:]))


print("Number of processors: ", mp.cpu_count())
cv2.useOptimized()
cv2.setNumThreads(mp.cpu_count())

model_path = 're_optimized_mbo_bisenetV10_pose106_fused_model_HWC.onnx'
#model_path = 're_optimized_mbo_bisenetV10_landmark_pose106_fused_model_HWC.onnx'

sess_options = ort.SessionOptions()
sess_options.inter_op_num_threads = 8
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

ort_session = ort.InferenceSession(model_path,
                                   providers=[
                                       ("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})],
                                   sess_options=sess_options)

# ort_session = ort.InferenceSession(model_path,
#                      providers=["CPUExecutionProvider"],
#                      sess_options=sess_options)
input_name = ort_session.get_inputs()[0].name
palette = generate_palette(11)
palette[0] = (0, 0, 0)
palette[1] = (0, 0, 0)

image_width = 640
image_height = 480
# 只取中間480*480縮放至192*192做推論，推論完再縮放回來
#  210        540              210
# |----|-------------|-----|

from tqdm import tqdm

base_videos = glob.glob('D:/downloads/Video/*.webm')
time_suffix = get_time_suffix()
make_dir_if_need(os.path.join('C:/Users/Allan/OneDrive/Documents/test_video', time_suffix))

tri = unpickle('./weights/tri.pkl')


# 獲取影像來源
# 如果想要改成webcam 改成  cv2.VideoCapture(0)即可
cap = cv2.VideoCapture(0)  # capture from camera
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 寫入成VIDEO  (如果沒有要錄製的話可以註解掉 設定為每秒20禎  720*720)
# vw = cv2.VideoWriter(os.path.join('C:/Users/Allan/OneDrive/Documents/test_video/', time_suffix, filename + '.avi'),
#                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (540, 960))

# rescale=rescale(1280/224.0,order=1)

hair_colors_imgs=glob.glob('./hair/*.jpg')
hair_colors_img=random.choice(hair_colors_imgs)
print(hair_colors_img)
hair_colors_hsv=cv2.cvtColor(image2array(hair_colors_img).astype(np.float32), cv2.COLOR_RGB2HSV)
hair_hsv_str=np.array(list(zip(hair_colors_hsv.reshape((-1, 3)).mean(0), hair_colors_hsv.reshape((-1, 3)).std(0), hair_colors_hsv.reshape((-1, 3)).min(0), hair_colors_hsv.reshape((-1, 3)).max(0))))
img_hsv_str=[]






scale = 192 / 480
yaw_stable = None
pitch_stable = None
roll_stable = None

infer_times=[]

frames_cnt=0
sum = 0
x_shift=0
y_shift=0
offsetx=0
offsety=0
process_time=[]
current_hair_color=''
i=0
while True:
    # 開始讀取影像
    if i > 0 and i % 180 == 0:
        hair_colors_img = random.choice(hair_colors_imgs)
        print(hair_colors_img)
        hair_colors_hsv = cv2.cvtColor(image2array(hair_colors_img).astype(np.float32), cv2.COLOR_RGB2HSV)
        hair_hsv_str = np.array(list(
            zip(hair_colors_hsv.reshape((-1, 3)).mean(0), hair_colors_hsv.reshape((-1, 3)).std(0),
                hair_colors_hsv.reshape((-1, 3)).min(0), hair_colors_hsv.reshape((-1, 3)).max(0))))
    ret, bgr_image = cap.read()
    H,W,C=bgr_image.shape
    bgr_image=bgr_image.astype(np.float32)
    scale = 192 / builtins.min(H,W)
    if H>W:
        offsety=(H-W)//2
    elif W>H:
        offsetx=(W-H)//2
    i+=1
    orig_bgr_image=bgr_image.copy()
    if bgr_image is None:
        print("no img")
        # 關閉VIDEO寫入
        #vw.release()
        break

    try:

        # 開始記錄時間
        st = time.time()
        # BGR轉RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # 縮放至192
        rgb_image = (crop_resize(rgb_image, offsetx=offsetx+x_shift, offsety=offsety+y_shift , cropx= builtins.min(H,W), cropy= builtins.min(H,W), tox=192, toy=192) - 127.5) / 127.5


        # 加入批次維   image_backend_adaptive轉成通道在前pytorch模式
        rgb_image = np.expand_dims(rgb_image, 0).astype(np.float32)

        # 開始記錄時間
        time_time = time.time()
        # 執行推論
        # landmarks, pose, facemesh, facecorner, cameramatrix = ort_session.run([], {input_name: rgb_image})
        hair_area_mask, alpha, landmarks, pose, facemesh, facecorner, cameramatrix = ort_session.run([], {
            input_name: rgb_image})

        process_time.append( time.time()-st)

        confidence = facecorner[0, 4]


        facecorner = facecorner[0, :4]
        facecorner = (facecorner / scale)
        landmarks =((landmarks[0]) / scale+0.5).astype(np.int32)
        facemesh = ((facemesh[0]) /scale+0.5).astype(np.int32)

        alpha = cv2.resize(alpha[0][0] * 255, None, fx=1.0 / scale, fy=1.0 / scale,
                           interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        alpha[alpha > 0.9] = 1
        alpha[alpha < 0.5] = 0
        hair_area_mask = cv2.resize(hair_area_mask[0][0], None, fx=1 / scale, fy=1 / scale,
                                    interpolation=cv2.INTER_NEAREST)

        headtop = facemesh[:, 1].min()
        headbtm = facemesh[:, 1].max()
        headleft = facemesh[:, 0].min()
        headright = facemesh[:, 0].max()

        facecorner[0] += offsetx + x_shift
        facecorner[2] += offsetx + x_shift
        facecorner[1] += offsety + y_shift
        facecorner[3] += offsety + y_shift
        landmarks[:, 0] += offsetx + x_shift
        landmarks[:, 1] += offsety + y_shift
        facemesh[:, 0] += offsetx + x_shift
        facemesh[:, 1] += offsety + y_shift




        pose = pose[0]

        this_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        hair_base = cv2.cvtColor(this_img.copy(), cv2.COLOR_RGB2HSV).astype(np.float32)
        bg_alpha = np.zeros_like(hair_base, dtype=np.float32)[:, :, 0]

        if offsetx>0:
            bg_alpha[:builtins.min(alpha.shape[0], bg_alpha.shape[0]),offsetx+x_shift:builtins.min(offsetx+x_shift+alpha.shape[1], bg_alpha.shape[1])] = \
                alpha#[:builtins.min(alpha.shape[0],bg_alpha.shape[0]),:builtins.min(offsetx+alpha.shape[1], bg_alpha.shape[1])-offsetx]
        alpha = bg_alpha

        hair_only_image = this_img * np.expand_dims(alpha,
                                                    -1)  # *(np.less_equal(np.expand_dims(this_gray,-1),100).astype(np.float32))
        matting_image = np.stack([bg_alpha * 0.5, bg_alpha, bg_alpha], axis=-1)
        resize_image = matting_image * 255 + bgr_image.copy() * (1 - matting_image)
        bk_image = (bgr_image.copy() + resize_image) / 2


        bg_hair_mask = np.zeros_like(hair_base, dtype=np.float32)[:, :, 0]
        bg_hair_mask[:builtins.min(hair_area_mask.shape[0], bg_hair_mask.shape[0]),
        offsetx:builtins.min(offsetx+hair_area_mask.shape[1], bg_hair_mask.shape[1])] = hair_area_mask[:builtins.min(hair_area_mask.shape[0],
                                                                                bg_hair_mask.shape[0]) ,
                                                            :builtins.min(offsetx+hair_area_mask.shape[1], bg_hair_mask.shape[1])-offsetx]
        hair_area_mask = bg_hair_mask

        hair_rgb_mean = bk_image.copy()[alpha == 1, :].mean()
        hair_rgb_std = np.sqrt(
            ((bk_image.copy()[alpha == 1, :] - bk_image.copy()[alpha == 1, :].mean(-1, keepdims=True)) ** 2).mean())
        base_image = bk_image.copy()


        hair_hsv = cv2.cvtColor(base_image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hair_rgb_base = base_image[alpha == 1, :]
        hair_base = hair_hsv[alpha == 1, :]
        if hair_base.size>0:
            img_hsv_str = np.array(list(zip(hair_base.reshape((-1, 3)).mean(0), hair_base.reshape((-1, 3)).std(0),
                                            hair_base.reshape((-1, 3)).min(0), hair_base.reshape((-1, 3)).max(0))))
            changed, changed_hsv_str = chage_color(hair_hsv, alpha, img_hsv_str, hair_hsv_str)
            # plt.imshow(changed/255.0)
            # plt.axis('off')
            # plt.show()
            # plt.close()

            hair_parsing = np.expand_dims(alpha, -1)
            # merged_image=base_image
            merged_image = (1 - hair_parsing) * base_image.copy() + hair_parsing * changed
            bgr_image = merged_image



        cv2.putText(bgr_image, "Confidence: {0:.1%}".format(confidence), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(180, 72, 36), 2, cv2.LINE_AA)
        fps=1.0/np.array(process_time)[-builtins.min(len(process_time),10):].mean()
        cv2.putText(bgr_image, "fps: {0:.2f}".format(fps), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 72, 36), 2, cv2.LINE_AA)

        print( "fps: {0:.2f}".format(fps),flush=True)
        if len(process_time)>50:
            process_time=process_time[-20:]

        # cv2.putText(bgr_image,  "offsetx: {0:.1f}  x_shift: {1:.1f} ".format(offsetx,x_shift), (10, 210), cv2.FONT_HERSHEY_SIMPLEX,0.7,(24, 72, 120), 2, cv2.LINE_AA)
        # cv2.putText(bgr_image, "offsety: {0:.1f}   y_shift: {1:.1f}".format(offsety,y_shift) , (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(24,72, 120), 2, cv2.LINE_AA)


        if confidence > 0.6:
            yaw_stable = 0.75 * yaw_stable + 0.25 * pose[0] if yaw_stable is not None else pose[0]
            pitch_stable = 0.5 * pitch_stable + 0.5 * pose[1] if pitch_stable is not None else pose[1]
            roll_stable = 0.75 * roll_stable + 0.25 * pose[2] if roll_stable is not None else pose[2]

            cv2.putText(bgr_image,  "Yaw: {0:.1f} ".format(yaw_stable), (10, 120), cv2.FONT_HERSHEY_SIMPLEX,0.7,(245, 22, 82), 1, cv2.LINE_AA)
            cv2.putText(bgr_image, "Pitch: {0:.1f}".format(pitch_stable) , (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(245, 22, 82), 1, cv2.LINE_AA)
            cv2.putText(bgr_image, "Roll: {0:.1f}".format(roll_stable), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(245, 22, 82), 1, cv2.LINE_AA)


            # bgr_image = cv2.rectangle(bgr_image, (int(facecorner[0]), int(facecorner[1])),
            #                           (int(facecorner[2]), int(facecorner[3])), color=(128, 256, 128), thickness=2)
            # bgr_image = cv2.rectangle(bgr_image, (int(offsetx + x_shift), int(offsety + y_shift)), (int(offsetx + x_shift) + builtins.min(H, W), int(offsety + y_shift) + builtins.min(H, W)),color=(128, 128, 256), thickness=2)


            bgr_image = cv2.addWeighted(np.clip(bgr_image, 0, 255).astype(np.uint8), 0.7, bgr_image.astype(np.uint8),
                                        0.3, 0)

            for k in range(len(facemesh)):
                (x, y, z) = facemesh[k]
                cv2.circle(bgr_image, (x, y), 2, (50, 205, 50), -1)
            cv2.polylines(bgr_image, pts=[landmarks[RIGHT_EYE_POINTS, :2].reshape((-1, 2)).astype(np.int32)],
                          isClosed=True, color=(255, 242, 0), thickness=1)
            cv2.polylines(bgr_image, pts=[landmarks[LEFT_EYE_POINTS, :2].reshape((-1, 2)).astype(np.int32)],
                          isClosed=True, color=(255, 242, 0), thickness=1)
            cv2.circle(bgr_image, (int(landmarks[104, 0]), int(landmarks[104, 1])), 2, (0, 255, 240), -1)
            cv2.circle(bgr_image, (int(landmarks[105, 0]), int(landmarks[105, 1])), 2, (0, 255, 240), -1)

            offset_ear=0.3*((landmarks[4]+ landmarks[28])/2-landmarks[53])
            for k in [4,28]:

                (x, y, z) = landmarks[k]+offset_ear
                cv2.circle(bgr_image, (int(x), int(y)), 5, (192, 128, 32), -1)

        if offsety > 0:
            if (headtop - (headbtm-headtop)*1.2) <20:
                y_shift =builtins.max(y_shift+ int((headtop - (headbtm-headtop)*1.2)-20),-offsety)
            elif builtins.min(H, W)-(headbtm +(headbtm-headtop)*0.5)<20:
                y_shift =builtins.min(y_shift+ int((builtins.min(H, W)-(headbtm +(headbtm-headtop)*0.5))-20),offsety)
            else:
                y_shift = int(0)
        elif offsetx > 0:
            if headleft < 20:
                x_shift = builtins.max(x_shift+int(headleft-20),-offsetx)
            elif builtins.min(H, W)-headright<20:
                x_shift =builtins.min(x_shift +int(20-(builtins.min(H, W)-headright)),offsetx)
            else:
                x_shift = int(0)
    except Exception as e:
        x_shift=0
        y_shift=0
        print(e)
        PrintException()


    # 寫入禎
    #vw.write(bgr_image)

    # 顯示禎
    frames_cnt += 1
    cv2.imshow('annotated', np.clip(bgr_image,0,255).astype(np.uint8))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 關閉影片數據
cap.release()
# 銷毀所有視窗
cv2.destroyAllWindows()
#vw.release()
