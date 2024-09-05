import numpy as np
import torch
import cv2

class Helpers:

    def __init__(self) -> None:
        pass

    def getCenterofBoundingBox(self,box):
        """
        
        """
        minx,miny,maxx,maxy = box
        pointxc = minx  + int(((maxx-minx))/2)
        pointyc = miny + int(((maxy-miny))/2)
        return pointxc,pointyc
    
    def calculate_iou(self, gt_mask, pred_mask, class_label=1):
        pred_mask = (pred_mask == class_label) * 1
        gt_mask = (gt_mask == class_label) * 1
        overlap = pred_mask * gt_mask  # Logical AND
        union = (pred_mask + gt_mask)>0  # Logical OR
        iou = overlap.sum() / float(union.sum())
        return iou
    

    def calcMask(self,model,filepath, input, gt_mask, box=None, points=None, class_label=255, input_label=1):
        if box is None and points is None:
            return None
        
        if box is not None:
            boxes = np.array(box).ravel()
            boxes = np.array(boxes)
            boxes = torch.tensor(boxes, device=model.device)
            boxes = model.transform.apply_boxes_torch(boxes, input.shape[:2])
        
        if points is not None: 
            points = np.array(points).ravel()
            points = np.array([points])

        input_label = np.array([input_label])

        try:
            masks, scores, logits = model.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=boxes,
                    multimask_output=True,
                )
            
            masks_combined = None
            for i, (mask,score) in enumerate(zip(masks,scores)):
                    score_ = score.cpu().numpy()
                    mask_ = mask.cpu().numpy()

                    for i,mask__ in enumerate(mask_):
                        h, w = mask_.shape[-2:]

                        if masks_combined is None:
                            masks_combined = np.zeros((h,w,1), dtype=np.uint8)

                        if score_[i]>0.8:
                            mask__ = mask__[:,:].reshape(h, w, 1).astype(np.uint8)
                            masks_combined = cv2.add(masks_combined, mask__)
            masks_combined *= 255

            masks_combined = cv2.threshold(masks_combined,1, 255, cv2.THRESH_BINARY)[1]
            IOU = self.calculate_iou(gt_mask,masks_combined, class_label=255)
            return IOU  
        
        except Exception as e:
            print(f"Failed to calculate IOU for: {filepath}")

        return None