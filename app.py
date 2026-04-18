
import gradio as gr
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import os
from torchinfo import summary
import albumentations as A
from albumentations.pytorch import ToTensorV2

import gspread
from google.oauth2.service_account import Credentials
import datetime

#Modal.com


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_model(checkpoint_path):
  model = smp.Unet(
    encoder_name="tu-convnext_tiny",
    encoder_weights=None, 
    in_channels=1,
    classes=1,
    activation=None, 
    decoder_attention_type='scse'
  )

  model.load_state_dict(torch.load(checkpoint_path, map_location=device))
  model.to(device) 
  model.eval()
  return model


def load_model2(checkpoint_path):

  model = smp.Unet(
     v       
      encoder_weights=None,    
      in_channels=1,                
      classes=1,                    
      activation=None,               
      decoder_attention_type='scse',
  ).to(device)

  state_dict = torch.load(checkpoint_path, map_location=device)

  from collections import OrderedDict
  new_state_dict = OrderedDict()
  for k, v in state_dict.items():
      name = k.replace('_orig_mod.', '')
      new_state_dict[name] = v

  model.load_state_dict(new_state_dict)

  model.eval()
  return model

kidney_model = load_model2("kidney_unet6.pt")
vessel_model = load_model("vessels_unet5.pt")


def pad_to_multiple(img, multiple=32, value=0):
  h, w = img.shape[:2]
  pad_h = (multiple - h % multiple) % multiple
  pad_w = (multiple - w % multiple) % multiple
  top = pad_h // 2
  bottom = pad_h - top
  left = pad_w // 2
  right = pad_w - left
  return cv2.copyMakeBorder(img, top, bottom, left, right, 
                borderType=cv2.BORDER_CONSTANT, value=value)

            

IMAGENET_MEAN = (0.485,)
IMAGENET_STD = (0.229,)

kidney_transform = A.Compose([
  A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
  A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ToTensorV2(),
])

vessel_transform = A.Compose([
  A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
  A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ToTensorV2(),
])

def log_to_google_sheet(k_centers, v_centers):
  try:
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file("google_key.json", scopes=scope)
    client = gspread.authorize(creds)
    

    sheet = client.open("kvultra").sheet1

    if not sheet.acell('A1').value:
      headers = ["Timestamp", "Kidney Centroid", "Vessel Centroid", "Kidney Centroid Millimeteres", "Vessel Centroid Millimeters"]
      sheet.insert_row(headers, index=1, value_input_option="RAW")
      sheet.format("A1:C1", {"textFormat": {"bold": True}})


    if k_centers:
      k_x, k_y = k_centers[0]
      k_centermm = (round(k_x * 0.41, 2), round(k_y * 0.41, 2))
    else:
      k_centermm = "N/A"

    if v_centers:
      v_x, v_y = v_centers[0]
      v_centermm = (round(v_x * 0.41, 2), round(v_y * 0.41, 2))
    else:
      v_centermm = "N/A"

    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [timestamp, str(k_centers), str(v_centers), str(k_centermm), str(v_centermm)]
    
    sheet.append_row(row)
  except Exception as e:
    print(f"Failed to log to Google Sheets: {e}")


def predict(input_img):
  gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
  
  gray_padded = pad_to_multiple(gray, 32)
  img_for_aug = np.expand_dims(gray_padded, axis=-1)
  
  k_transformed = kidney_transform(image=img_for_aug)
  k_tensor = k_transformed["image"].unsqueeze(0).to(device)

  v_transformed = vessel_transform(image=img_for_aug)
  v_tensor = v_transformed["image"].unsqueeze(0).to(device)

  with torch.no_grad():
    k_logits = kidney_model(k_tensor)
    v_logits = vessel_model(v_tensor)
    
    k_probs = torch.sigmoid(k_logits)[0, 0].cpu().numpy()
    v_probs = torch.sigmoid(v_logits)[0, 0].cpu().numpy()

  k_mask = (k_probs > 0.5).astype(np.uint8)
  v_mask = (v_probs > 0.5).astype(np.uint8) 

  res_img = cv2.cvtColor(gray_padded, cv2.COLOR_GRAY2RGB)
  final_output = res_img.copy()

  kidney_layer = res_img.copy()
  kidney_layer[k_mask == 1] = [0, 255, 0] 

  alpha_k = 0.15 
  final_output = cv2.addWeighted(kidney_layer, alpha_k, final_output, 1 - alpha_k, 0)

  vessel_layer = final_output.copy()
  vessel_layer[v_mask == 1] = [255, 0, 0] 

  alpha_v = 0.7
  final_output = cv2.addWeighted(vessel_layer, alpha_v, final_output, 1 - alpha_v, 0)
  
  k_centers_list = []
  v_centers_list = []

  num_labels_k, _, _, centers_k = cv2.connectedComponentsWithStats(k_mask)
  for i in range(1, num_labels_k):
    cx, cy = int(centers_k[i][0]), int(centers_k[i][1])
    k_centers_list.append((cx, cy))

  num_labels_v, _, _, centers_v = cv2.connectedComponentsWithStats(v_mask)
  for i in range(1, num_labels_v):
    cx, cy = int(centers_v[i][0]), int(centers_v[i][1])
    v_centers_list.append((cx, cy))

  def get_main_centroid(mask, min_area):
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
      areas = stats[1:, cv2.CC_STAT_AREA]
      max_idx = np.argmax(areas) + 1
      
      if stats[max_idx, cv2.CC_STAT_AREA] > min_area:
        return [ (int(centers[max_idx][0]), int(centers[max_idx][1])) ]
    return []

  k_centers_list = get_main_centroid(k_mask, 50)
  v_centers_list = get_main_centroid(v_mask, 20)

  for cx, cy in k_centers_list:
    cv2.circle(final_output, (cx, cy), 5, (255, 255, 255), -1)
    cv2.putText(final_output, f"K: {cx},{cy}", (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

  for cx, cy in v_centers_list:
    cv2.circle(final_output, (cx, cy), 4, (255, 255, 255), -1)
    cv2.putText(final_output, f"V: {cx},{cy}", (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

  k_centers_list = []
  v_centers_list = []

  num_labels_k, _, stats_k, centers_k = cv2.connectedComponentsWithStats(k_mask)
  for i in range(1, num_labels_k):
    if stats_k[i, cv2.CC_STAT_AREA] > 50: 
      cx, cy = int(centers_k[i][0]), int(centers_k[i][1])
      k_centers_list.append((cx, cy))
      

  num_labels_v, _, stats_v, centers_v = cv2.connectedComponentsWithStats(v_mask)
  for i in range(1, num_labels_v):
    if stats_v[i, cv2.CC_STAT_AREA] > 20: 
      cx, cy = int(centers_v[i][0]), int(centers_v[i][1])
      v_centers_list.append((cx, cy))
      

  log_to_google_sheet(k_centers_list, v_centers_list)

  return final_output

demo = gr.Interface(
  fn=predict, 
  inputs=gr.Image(label="Upload Ultrasound Image"), 
  outputs=gr.Image(label="Kidney (Green) & Vessels (Red)"),
  title="Kidney Vessel AI Locator"
)


demo.launch(share=True)

"""
kernel = np.ones((3,3), np.uint8)
# Fills small holes inside the vessel
v_mask = cv2.morphologyEx(v_mask, cv2.MORPH_CLOSE, kernel)
"""