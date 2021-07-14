# 27brain's function  codes

import os
import json
import shutil
from pathlib import Path
import pandas as pd
import glob
from tqdm import tqdm 
import multiprocessing.dummy as mp 
from itertools import repeat
import pydicom
import numpy as np
import png


def dcm_dir_to_png_dir_SLICE(src_dir,
                            dst_dir,
                            csv_fp,
                            boat,
                            cohort = 'T2w',
                            slice_num = 0.5 , 
                            worker = 16):

  print("Building png from dcm...")
  train_df = pd.read_csv(csv_fp)
  brats21id_list = train_df["BraTS21ID"].tolist()

  fp_list = []
  for i, brats21id in enumerate(brats21id_list):
    # print(i,brats21id)
    patient_path = os.path.join(
        src_dir, 
        str(brats21id).zfill(5),
    )
    dcm_fp_list_SORTED = sorted(
          glob.glob(os.path.join(patient_path, cohort, "*")), 
          key=lambda x: int(x[:-4].split("-")[-1]),
    )
    target_fp = dcm_fp_list_SORTED[int(len(dcm_fp_list_SORTED) * slice_num)]
    fp_list += [target_fp]
  # print (fp_list)

  # stop()

  dst_fp_list = []
  for fp in fp_list:
    if fp.endswith(".dcm"):
        fn = os.path.basename(fp)
        cut_fn = os.path.splitext(fn)[0]
        dst_fn = cut_fn + '.png'
        dst_fp = os.path.join(dst_dir, dst_fn) 
        dst_fp_list += [dst_fp]

  
  pbar = tqdm(total=len(fp_list), position=0, leave=True,
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
 
  p=mp.Pool(worker)
  p.starmap(dcm2png, zip(fp_list, dst_fp_list,repeat(boat),repeat(pbar)))
  p.close()
  pbar.close()
  p.join()


def dcm2png(dcm_filepath,png_filepath,boat,pbar = None):
  if boat.is_halt:
    return

  ds = pydicom.dcmread(dcm_filepath)

  # try:
  #   shape = ds.pixel_array.shape
  # except Exception as e:
  #   print('error occured: ' + str(e))
  #   return
  shape = ds.pixel_array.shape

  # Convert to float to avoid overflow or underflow losses.
  image_2d = ds.pixel_array.astype(float)

  # Rescaling grey scale between 0-255
  image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
  image_2d_scaled = np.uint8(image_2d_scaled)  # Convert to uint
  
  if pbar:
    pbar.update(1)
    
  # Write the PNG file
  with open(png_filepath, 'wb') as png_file:
    w = png.Writer(shape[1], shape[0], greyscale=True)
    w.write(png_file, image_2d_scaled)



def stop(msg= 'stopped'):
  print("\n")
  raise Exception(msg)

def mkdir(dirname):
  Path(dirname).mkdir(parents=True, exist_ok=True)

 


def empty_dir(dirname):
  folder = dirname
  for filename in os.listdir(folder):
      file_path = os.path.join(folder, filename)
      try:
          if os.path.isfile(file_path) or os.path.islink(file_path):
              os.unlink(file_path)
          elif os.path.isdir(file_path):
              shutil.rmtree(file_path)
      except Exception as e:
          print('Failed to delete %s. Reason: %s' % (file_path, e))






def make_submit_dict(cut_fn_list,predict_list,dcm_data_dict):
  # print(cut_fn_list)
  # print(predict_list)

  # make predict_dict
  predict_dict = {}
  for i,val in enumerate(predict_list):
      
      if val['image_id'] in predict_dict:
          predict_dict[val['image_id']] += [val]
      else:
          predict_dict[val['image_id']] = [val] 

  # make study_dict
  study_dict = {}
  for img_id,val in dcm_data_dict.items():
      # print(img_id, val)
      if val['study_id'] in study_dict:
          study_dict[val['study_id']] += [val]
      else:
          study_dict[val['study_id']] = [val] 
  
  print(study_dict)



  to_submit = []

  # make study line
  for study_id,study_list in study_dict.items():
    # print(len(study_list))
    s_item = {}
    s_item['id'] = study_id+ "_study"
    s_item['PredictionString']="negative 1 0 0 1 1"
    to_submit+=[s_item]

  # make image line
  for cut_fn in cut_fn_list:
    s_item = {}
    s_item['id'] = cut_fn+ "_image"

    if cut_fn in predict_dict:
      crop_str_list = []
      for crop in predict_dict[cut_fn]:
        x,y,w,h = crop['bbox'][0],crop['bbox'][1],crop['bbox'][2],crop['bbox'][3]
        crop_str = 'opacity {} {} {} {} {}'.format(
          crop['score'],int(x),int(y),int(x+w),int(y+h)
        )
        crop_str_list += [crop_str]
      s_item['PredictionString']=' '.join(crop_str_list)
    else:
      s_item['PredictionString']="none 1 0 0 1 1"
    
    to_submit+=[s_item]
  
  return to_submit





def read_json(fp):
  with open(fp) as f:
      d = json.load(f)
  
  return(d)

def get_dcmFilePathList(dcm_dir):
  dcm_filepath_list =[]
  for root, dirs, files in os.walk(dcm_dir):
    for file in files:
      if file.endswith(".dcm"):
        filepath = os.path.join(root, file)
        dcm_filepath_list += [filepath]
  return dcm_filepath_list


def get_cut_fn_list(dir_name):
  fp_list = get_dcmFilePathList(dir_name)
  cut_fn_list = []
  for fp in fp_list:
    cut_fn = os.path.splitext(os.path.basename(fp))[0]
    cut_fn_list += [cut_fn]

  return cut_fn_list





