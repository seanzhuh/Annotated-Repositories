import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

image = Image.open('notebooks/images/truck.jpg')
image = np.array(image.convert("RGB"))

input_point = np.array([[500, 375]])
input_label = np.array([1])

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    # masks, _, _ = predictor.predict(
    #     point_coords=input_point,
    #     point_labels=input_label,
    #     multimask_output=False
    # )
    # masks, scores, _ = predictor.predict(
    #     point_coords=None,
    #     point_labels=None,
    #     box=input_boxes,
    #     multimask_output=False,
    # )
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    