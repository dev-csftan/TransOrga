import os 
import cv2
from SegTracker_our import SegTracker
from PIL import Image
from model_args import aot_args, segtracker_args
from aot_tracker import _palette
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import gc

def num_mask(frame, mask):
    
    cell_mask = np.zeros((mask.shape[0],mask.shape[1],3))
    cell_num = len(np.unique(mask)) - 1

    properties = {}
    start_idx = 0

    for i in range(1, cell_num+1):
        mask_one = np.array(mask == np.unique(mask)[i],dtype=np.uint8)
        try:
            cell_color = (0,255,0)
            contours, hierarchy = cv2.findContours(mask_one, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(cell_mask, contours, -1, cell_color, 3)
            text = str(i+start_idx)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1 
            font_color = cell_color
            thickness = 3

            text_size, _ = cv2.getTextSize(text,font,font_scale, thickness)
            #calculate 
            text_x = ( np.max(np.where(mask_one==1)[0]) - np.min(np.where(mask_one==1)[0]))//2 + np.min(np.where(mask_one==1)[0])
            text_y = ( np.max(np.where(mask_one==1)[1]) - np.min(np.where(mask_one==1)[1]))//2 + np.min(np.where(mask_one==1)[1])

            cell_mask = cv2.putText(cell_mask, text, (text_y, text_x), font, font_scale, font_color, thickness)
        
        except ZeroDivisionError:
            pass
    cell_mask =cv2.addWeighted(np.array(cell_mask, dtype=np.uint8), 1, frame, 1, 0)

    return cell_mask


def main(input_path, output_gif,output_video):

    cap =cv2.VideoCapture(input_path)
    fsp = cap.get(cv2.CAP_PROP_FPS)
    pred_list =[]
    masked_pred_list = []
    torch.cuda.empty_cache()
    gc.collect()
    track_gap = segtracker_args['track_gap']
    frame_idx = 0
    segtracker = SegTracker(segtracker_args,aot_args)
    segtracker.restart_tracker()

    with torch.cuda.amp.autocast():
        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            if frame_idx ==0:
                pred_mask =segtracker.seg(frame)
                torch.cuda.empty_cache()
                gc.collect()
                segtracker.add_reference(frame, pred_mask)
            elif (frame_idx % track_gap) ==0:
                seg_mask = segtracker.seg(frame)
                torch.cuda.empty_cache()
                gc.collect()
                track_mask = segtracker.track(frame)

                new_obj_mask = segtracker.find_new_objs(track_mask,seg_mask)

                pred_mask = track_mask + new_obj_mask

                segtracker.add_reference(frame, pred_mask)

            else:
                pred_mask = segtracker.track(frame, update_memory=True)
            torch.cuda.empty_cache()
            gc.collect()

            pred_list.append(pred_mask)
            frame_idx +=1
        cap.release()

    cap =cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    width =int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if input_path[-3:]=='mp4':
        fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
    elif input_path[-3:] == 'avi':
        fourcc =  cv2.VideoWriter_fourcc(*"MJPG")
        # fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        #fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame =cap.read()
        if not ret:
            break
        frame =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred_mask = pred_list[frame_idx]
        masked_frame = num_mask(frame, pred_mask)
        masked_frame =cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)
        out.write(masked_frame)
        frame_idx +=1
    out.release()
    cap.release()
    imageio.mimsave(output_gif,pred_list,duration=(1000 * 1/fps))
    del segtracker
    torch.cuda.empty_cache()
    gc.collect()

if __name__=='__main__':
    input_path = '20210518_U049MIS002_XY007_Z3_C1.gif'
    output_gif = 'org_seg_original.gif'
    output_video = 'org_seg_original.mp4'

    main(input_path,output_gif,output_video)




