# Copyright 2025 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2025/11/24 ImageMaskDatasetGenerator.py


import os
import sys
import io
import shutil
import glob
import nibabel as nib
import numpy as np
from PIL import Image, ImageOps
import traceback
import math
from scipy.ndimage import map_coordinates

from scipy.ndimage import gaussian_filter
import cv2

class ImageMaskDatasetGenerator:

  def __init__(self, 
               images_dir  = "./", 
               masks_dir   = "./",
               output_dir = "./", 
               resize     = 512,
               angle      = cv2.ROTATE_90_COUNTERCLOCKWISE):
    
    self.images_dir = images_dir
    self.masks_dir  = masks_dir
    
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    self.output_images_dir = os.path.join(output_dir, "images")
    self.output_masks_dir  = os.path.join(output_dir, "masks")

    os.makedirs(self.output_images_dir)
    os.makedirs(self.output_masks_dir)
    
    self.image_files = glob.glob(self.images_dir + "/*_image.nii")
    self.image_files = sorted(self.image_files)

    self.mask_files = glob.glob(self.images_dir + "/*_label.nii")
    self.mask_files = sorted(self.mask_files)

    self.RESIZE    = (resize, resize)
    self.seed      = 137
    self.W         = resize
    self.H         = resize
    self.angle     = angle
    self.file_format= ".png"
  
  def generate(self):
    index = 10000
 
    num_mask_files  = len(self.mask_files)
 
    print("num_mask_files {}".format(num_mask_files))

    for i in range(num_mask_files):
      index +=1
      
      mask_file  = self.mask_files[i]
      #basename = os.path.basename(mask_file)
      image_file = self.image_files[i]
      self.generate_mask_files(mask_file,   index) 

      self.generate_image_files(image_file, index) 

  def resize_to_square(self, image, mask=True):
    #image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    RESIZE = h
    if w > h:
      RESIZE = w
    # 1. Create a black background
    if mask:
      background = np.zeros((RESIZE, RESIZE, 3),  np.uint8) 
    else:
      pixel = image[10][10]
      background = np.ones((RESIZE, RESIZE, 3),  np.uint8) * pixel 
    x = int((RESIZE - w)/2)
    y = int((RESIZE - h)/2)
    # 2. Paste the image to the background 
    background[y:y+h, x:x+w] = image
    # 3. Resize the background to (512x512)
    resized = cv2.resize(background, (self.W, self.H), cv2.INTER_LANCZOS4)

    return resized

  def normalize(self, image):
    min = np.min(image)/255.0
    max = np.max(image)/255.0
    scale = (max - min)
    if scale == 0:
      scale +=  1
    image = (image -min) / scale
    image = image.astype('uint8') 
    return image
  
  def colorize_mask(self, mask):
    h, w = mask.shape[:2]
    colorized = np.zeros((h, w, 3), dtype=np.uint8)

    colorized[np.equal(mask, 1)] = (255,0,255)    # AV:  mazenda
    colorized[np.equal(mask, 2)] = (255,255, 0)   # MV:  cyan
    colorized[np.equal(mask, 3)] = (255,0,0)      # AO:  blue 
    colorized[np.equal(mask, 4)] = (0,255,0)      # LA:  green
    colorized[np.equal(mask, 5)] = (0,255,255)    # LV:  yellow
  
    colorized[np.equal(mask, 6)] = (0,0,255)      # Myocardium : red
    colorized[np.equal(mask, 7)] = (110,110,110)  # Excised myocardium:   dark gray 
    
    return colorized

  # Modified to save plt-image to BytesIO() not to a file.
  def generate_image_files(self, niigz_file, index):
    nii = nib.load(niigz_file)
    fdata  = nii.get_fdata()
   
    w, h, d = fdata.shape
    print("=== image shape {}".format(fdata.shape))
    for i in range(d):
      img = fdata[:,:, i]
      filename  = str(index) + "_" + str(i) + self.file_format
      filepath  = os.path.join(self.output_images_dir, filename)
      corresponding_mask_file = os.path.join(self.output_masks_dir, filename)
      
      if os.path.exists(corresponding_mask_file):
    
        img   = self.normalize(img)   
        img  = img.astype('uint8') 
        img  = cv2.resize(img, self.RESIZE)
        img  = cv2.rotate(img, self.angle)
        cv2.imwrite(filepath, img)
        print("=== Saved {}".format(filepath))

      else:
        print("=== Skipped image {}".format(filepath))

  def generate_mask_files(self, niigz_file, index ):
    nii = nib.load(niigz_file)
    fdata  = nii.get_fdata()
   
    w, h, d = fdata.shape
    print("=== mask shape {}".format(fdata.shape))
    #input("----HIT any key")
    for i in range(d):
      img = fdata[:,:, i]
      filename  = str(index) + "_" + str(i) + self.file_format
      filepath  = os.path.join(self.output_masks_dir, filename)
      
      if img.any() >0:
        #img = img.astype('uint8')
        img  = self.colorize_mask(img)
        img  = cv2.resize(img, self.RESIZE)
        img  = cv2.rotate(img, self.angle)

        print("--- Saved {}".format(filepath))
        cv2.imwrite(filepath, img)

      else:
        print("=== Skipped mask file{}".format(filepath))


if __name__ == "__main__":
  try:

    images_dir  = "./HOCMvalvesSeg/"
    masks_dir   = "./HOCMvalvesSeg/"
    
    output_dir  = "./HOCMvalves-master/"
    angle       = cv2.ROTATE_90_COUNTERCLOCKWISE
    generator = ImageMaskDatasetGenerator(images_dir  = images_dir, 
                                          masks_dir  = masks_dir,
                                          output_dir = output_dir, 
                                          angle      = angle)
    generator.generate()
  except:
    traceback.print_exc()

 
