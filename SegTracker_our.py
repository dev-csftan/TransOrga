import sys
sys.path.append("..")
sys.path.append("./sam")
from sam.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from aot_tracker import get_aot
import numpy as np
from tool.detector import Detector
from tool.transfer_tools import draw_outline, draw_points
import cv2
from seg_track_anything import draw_mask
import torch
import copy
from PIL import Image

import torchvision.transforms as transforms
class SegTracker():
    def __init__(self,segtracker_args, aot_args) -> None:
        """
         Initialize SAM and AOT.
        """
        self.device =segtracker_args['device']
        self.transorga = torch.load('checkpoint.pth', map_location='cpu') 
        self.transorga.to(self.device)
        self.transorga.eval()
        self.tracker = get_aot(aot_args)
        
        self.input_size = 512
        self.detector = Detector(self.device)
        #self.sam_gap = segtracker_args['sam_gap']
        self.min_area = segtracker_args['min_area']
        self.max_obj_num = segtracker_args['max_obj_num']
        self.min_new_obj_iou = segtracker_args['min_new_obj_iou']
        self.reference_objs_list = []
        self.object_idx = 1
        self.origin_merged_mask = None  # init by segment-everything or update
        self.first_frame_mask = None

        # debug
        self.everything_points = []
        self.everything_labels = []
        print("SegTracker has been initialized")

    
    def split_img(self,image):
        #image = Image.open(img_path).convert("RGB")
        ##
        pil_image = Image.fromarray(image)
        ##

        pil_image = pil_image.resize((1024,1024))
        sub_imgs =[]
        for row in range(3):
            for col in range(3):
                left = col *256
                upper = row *256
                right = left +512
                lower = upper+512

                sub_image =pil_image.crop((left,upper,right,lower))
                sub_imgs.append(sub_image)
        
        return sub_imgs

    def merge_image(self,images):
        merge_image =Image.new("L",[1024,1024])
        new_image_0_0 = images[0].crop((0,0,384,384))
        new_image_0_1 = images[1].crop((128,0,384,384))
        new_image_0_2 = images[2].crop((128,0,512,384))
        new_image_1_0 = images[3].crop((0,128,384,384))
        new_image_1_1 = images[4].crop((128,128,384,384))
        new_image_1_2 = images[5].crop((128,128,512,384))
        new_image_2_0 = images[6].crop((0,128,384,512))
        new_image_2_1 = images[7].crop((128,128,384,512))
        new_image_2_2 = images[8].crop((128,128,512,512))

        merge_image.paste(new_image_0_0,(0,0))
        merge_image.paste(new_image_0_1,(384,0))
        merge_image.paste(new_image_0_2,(640,0))
        merge_image.paste(new_image_1_0,(0,384))
        merge_image.paste(new_image_1_1,(384,384))
        merge_image.paste(new_image_1_2,(640,384))
        merge_image.paste(new_image_2_0,(0,640))
        merge_image.paste(new_image_2_1,(384,640))
        merge_image.paste(new_image_2_2,(640,640))

        return merge_image


    def remove_edge_cells(self, mask_image):
        w,h = mask_image.shape
        pruned_mask = copy.deepcopy(mask_image)
        remove_list = []
        edges = mask_image[0,:],mask_image[w-1,:],mask_image[:,0],mask_image[:,h-1]
        for edge in edges:
            edge_masks = np.unique(edge)
            for edge_mask in edge_masks:
                remove_list.append(edge_mask)
                pruned_mask[np.where(mask_image==edge_mask)] = 0

        return pruned_mask
    def remove_small_cells(self,mask_image,area_threshold=10):
        w,h = mask_image.shape
        pruned_mask = copy.deepcopy(mask_image)
        for mask_index in np.unique(mask_image):
            # if mask_index == mask_image[330,640]:
            #     a = 1
            area = np.sum(mask_image == mask_index)
            if area < area_threshold:
                pruned_mask[np.where(mask_image == mask_index)] = 0

        return pruned_mask
    def remove_concentric_masks(self,mask_image):
        # Convert the mask image to grayscale
        cell_values = np.unique(mask_image)
        for i in range(1, len(cell_values)):# remove background
            mask_one = np.array(mask_image == cell_values[i],dtype=np.uint8)
            # mask_one_dilated = cv2.dilate(mask_one, np.ones((5, 5), np.uint8),100)
            # xmin, xmax, ymin, ymax = np.min(np.where(mask_one == 1)[0]), np.max(np.where(mask_one == 1)[0]),\
            #     np.min(np.where(mask_one == 1)[1]), np.max(np.where(mask_one == 1)[1]),
            contour, _ = cv2.findContours(mask_one, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contour) > 0:
                largest_contour = max(contour, key=cv2.contourArea)

                mask_image = cv2.drawContours(mask_image, [largest_contour], -1, (int(cell_values[i])), thickness=cv2.FILLED)
        return mask_image

    def post_process(self, final_mask,frame):

        gray_image = final_mask.convert("L")
        image_array = np.array(gray_image)
        #print('frame',frame.shape)
        width, height = frame.shape[0], frame.shape[1]

        _,thresholded_image =cv2.threshold(image_array,0,255,cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_image, connectivity=8)
        output_image = np.zeros_like(gray_image)
        for label in range(1, num_labels):
            output_image[labels ==label] =label
        pruned_mask =self.remove_edge_cells(output_image)
        pruned_mask_reduce = self.remove_small_cells(pruned_mask,area_threshold=150)
        pruned_mask_reduce = self.remove_concentric_masks(pruned_mask_reduce)
        pruned_mask_reduce = np.resize(pruned_mask_reduce,(width,height))
        return pruned_mask_reduce

    def seg(self,frame):
        frames = self.split_img(frame)
        fs = []
        for f in frames:
            f = transforms.ToTensor()(f)
            f = f.to(self.device)
            f = f.view(1,3,512,512)
            prediction, attention =self.transorga(f)
            output = torch.argmax(prediction, dim=1)
            tensor_to_image = transforms.ToPILImage()
            output_img = tensor_to_image(output.to(torch.float32))
            fs.append(output_img)
        second_mask = self.merge_image(fs)
        final_mask = self.post_process(second_mask,frame)

        return final_mask
    
    def update_origin_merged_mask(self, updated_merged_mask):
        self.origin_merged_mask = updated_merged_mask
        self.object_idx += 1

    def reset_origin_merged_mask(self, mask, id):
        self.origin_merged_mask = mask
        self.object_idx = id

    def add_reference(self,frame,mask,frame_step=0):
        '''
        Add objects in a mask for tracking.
        Arguments:
            frame: numpy array (h,w,3)
            mask: numpy array (h,w)
        '''
        self.reference_objs_list.append(np.unique(mask))
        self.tracker.add_reference_frame(frame,mask,self.get_obj_num(),frame_step)
    
    def track(self,frame,update_memory=False):
        '''
        Track all known objects.
        Arguments:
            frame: numpy array (h,w,3)
        Return:
            origin_merged_mask: numpy array (h,w)
        '''
        pred_mask = self.tracker.track(frame)
        if update_memory:
            self.tracker.update_memory(pred_mask)
        return pred_mask.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    
    def get_tracking_objs(self):
        objs = set()
        for ref in self.reference_objs_list:
            objs.update(set(ref))
        objs = list(sorted(list(objs)))
        objs = [i for i in objs if i!=0]
        return objs
    
    def get_obj_num(self):
        return int(max(self.get_tracking_objs()))
    
    def find_new_objs(self, track_mask, seg_mask):
        '''
        Compare tracked results from AOT with segmented results from SAM. Select objects from background if they are not tracked.
        Arguments:
            track_mask: numpy array (h,w)
            seg_mask: numpy array (h,w)
        Return:
            new_obj_mask: numpy array (h,w)
        '''
        new_obj_mask = (track_mask==0) * seg_mask
        new_obj_ids = np.unique(new_obj_mask)
        new_obj_ids = new_obj_ids[new_obj_ids!=0]
        obj_num = self.get_obj_num() + 1
        for idx in new_obj_ids:
            new_obj_area = np.sum(new_obj_mask==idx)
            obj_area = np.sum(seg_mask==idx)
            if new_obj_area/obj_area < self.min_new_obj_iou or new_obj_area < self.min_area\
                or obj_num > self.max_obj_num:
                new_obj_mask[new_obj_mask==idx] = 0
            else:
                new_obj_mask[new_obj_mask==idx] = obj_num
                obj_num += 1
        return new_obj_mask
        
    def restart_tracker(self):
        self.tracker.restart()

    def seg_acc_bbox(self, origin_frame: np.ndarray, bbox: np.ndarray,):
        ''''
        Use bbox-prompt to get mask
        Parameters:
            origin_frame: H, W, C
            bbox: [[x0, y0], [x1, y1]]
        Return:
            refined_merged_mask: numpy array (h, w)
            masked_frame: numpy array (h, w, c)
        '''
        # get interactive_mask
        interactive_mask = self.sam.segment_with_box(origin_frame, bbox)[0]
        refined_merged_mask = self.add_mask(interactive_mask)

        # draw mask
        masked_frame = draw_mask(origin_frame.copy(), refined_merged_mask)

        # draw bbox
        masked_frame = cv2.rectangle(masked_frame, bbox[0], bbox[1], (0, 0, 255))

        return refined_merged_mask, masked_frame

    def seg_acc_click(self, origin_frame: np.ndarray, coords: np.ndarray, modes: np.ndarray, multimask=True):
        '''
        Use point-prompt to get mask
        Parameters:
            origin_frame: H, W, C
            coords: nd.array [[x, y]]
            modes: nd.array [[1]]
        Return:
            refined_merged_mask: numpy array (h, w)
            masked_frame: numpy array (h, w, c)
        '''
        # get interactive_mask
        interactive_mask = self.sam.segment_with_click(origin_frame, coords, modes, multimask)

        refined_merged_mask = self.add_mask(interactive_mask)

        # draw mask
        masked_frame = draw_mask(origin_frame.copy(), refined_merged_mask)

        # draw points
        # self.everything_labels = np.array(self.everything_labels).astype(np.int64)
        # self.everything_points = np.array(self.everything_points).astype(np.int64)

        masked_frame = draw_points(coords, modes, masked_frame)

        # draw outline
        masked_frame = draw_outline(interactive_mask, masked_frame)

        return refined_merged_mask, masked_frame

    def add_mask(self, interactive_mask: np.ndarray):
        '''
        Merge interactive mask with self.origin_merged_mask
        Parameters:
            interactive_mask: numpy array (h, w)
        Return:
            refined_merged_mask: numpy array (h, w)
        '''
        if self.origin_merged_mask is None:
            self.origin_merged_mask = np.zeros(interactive_mask.shape,dtype=np.uint8)

        refined_merged_mask = self.origin_merged_mask.copy()
        refined_merged_mask[interactive_mask > 0] = self.object_idx

        return refined_merged_mask
    
    def detect_and_seg(self, origin_frame: np.ndarray, grounding_caption, box_threshold, text_threshold):
        '''
        Using Grounding-DINO to detect object acc Text-prompts
        Retrun:
            refined_merged_mask: numpy array (h, w)
            annotated_frame: numpy array (h, w, 3)
        '''
        # backup id and origin-merged-mask
        bc_id = self.object_idx
        bc_mask = self.origin_merged_mask

        # get annotated_frame and boxes
        annotated_frame, boxes = self.detector.run_grounding(origin_frame, grounding_caption, box_threshold, text_threshold)
        for i in range(len(boxes)):
            bbox = boxes[i]
            interactive_mask = self.sam.segment_with_box(origin_frame, bbox)[0]
            refined_merged_mask = self.add_mask(interactive_mask)
            self.update_origin_merged_mask(refined_merged_mask)

        # reset origin_mask
        self.reset_origin_merged_mask(bc_mask, bc_id)

        return refined_merged_mask, annotated_frame
