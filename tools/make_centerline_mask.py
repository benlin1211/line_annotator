
import argparse
import glob, os
import numpy as np
# import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import time
#import cv2
from skimage.io import imread
from PIL import Image
import re
import csv
from tqdm import trange, tqdm

TOTAL_FRINGE=3 # 非定位圖、非顏色塗的數量：00,01,02 共3張
"""
    共48條線。每個線有512個離散v值所對應的 sub-pixe u值。
    在UV folder, debug-UV-xxxx.csv 表中code欄位 標註碼號
    Code 00+3n : 由 image 00 產生
    Code 01+3n : 由 image 01 產生
    Code 02+3n : 由 image 02 產生
    n = 0..47
    height欄位表示, uv detect結果 debug-CamWithUV-xxxx.png (u,v), v = height 時. u 浮點位置為 qq.qqq。
    debug-CamWithUV-xxxx.png 屬於化繁為整顯示。
    座標系：(u,v) =(0,0) 影像左上角, u橫軸由左向右遞增、v縱軸由上到下遞增。Code 編碼由左至右遞增。
"""

def extract_int_from_str(input_string:str):
    extracted_number = int(re.search(r'\d+', input_string).group())
    return extracted_number

def line_augmentation(input_image, lines, k):
    assert len(input_image.shape) == 2, "Expect input_image is HxW."
    # assert input_image.shape[2] == 1, "Expect input_image have 1 channel."
    centerline_images = np.zeros([input_image.shape[0], input_image.shape[1]], dtype=np.uint8)
    for line in lines:
        line_number = extract_int_from_str(line[0])
        # 去頭
        coor = line[1:]
        # v是高度(int), u是該高度v的對應u值(float)
        for v in range(len(coor)):
            # 似乎是向下取整 而不是四捨五入
            # u=round(float(line[v]))
            u=int(float(coor[v]))
            if u != -1.0 and line_number%TOTAL_FRINGE==k:
                #print(f"({u},{v})")
                # u、v分別由左向右、上到下遞增
                centerline_images[v][u] = 255

    enhance_image = np.copy(input_image)
    enhance_image[np.where(centerline_images!=0)] = centerline_images[np.where(centerline_images!=0)]

    return enhance_image, centerline_images # 255, 255, 1


def read_UV_csv(file_path):
    # 522 columns: 0: code=1/2/3+3n, 1-520: 該線心的sub-pixel v 座標（pixel）, 521: null(多的)
    # 145 rows: 0-143: 第幾張圖的第幾號線心(有3張圖, 每張圖48條),144: null(多的)
    # 數值意義: 第幾張圖的第幾號線心, 在第v個高度下的u值（sub-pixel）
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file) 
        # Skip the header (first row)
        next(csv_reader, None)
        for row in csv_reader:
            # 去尾 null
            data.append(row[:-1])
    # 去尾 null
    data=data[:-1]
    return data 

RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
ORANGE = (255,128,0)
CHARTREUSE = (128,255,0)
VIOLET = (128,0,255)
DEEP_PINK = (255,0,128)
SPRING_GREEN = (0,255,128)
DODGER_BLUE = (0,128,255)

def index_extractor(image, target_color):
    indices = np.where(np.all(image == target_color, axis=-1))
    return indices
    

def make_data(data_root, UV_root, save_path_UV):

     # find matching image pairs by UV_csv_names
    _fringe_names = sorted(glob.glob(os.path.join(data_root, "*.bmp"), recursive=True))

    # # fetch information from .csv in UV folder
    # # Split the path into its components
    # path_components = data_root.split(os.path.sep)
    # print(path_components)
    # # Remove the last directory
    # path_components.pop()
    # # Add the new directory to the path
    # path_components.append('UV')
    # # Join the components back into a path
    # UV_path = os.path.sep.join(path_components)
    # print(UV_path)

    UV_csv_names = sorted(glob.glob(os.path.join(UV_root, "*.csv"), recursive=True))
    scene_number = [name.split("-")[-1].replace(".csv","") for name in UV_csv_names]

    # Exclude image 03, 04, 05, 06, 07
    fringe_names = [name for name in _fringe_names if name.split("-")[-2] in scene_number and re.search(r'0[0-2].bmp', name)]
    # # expand csv names so that the length meets.
    # UV_csv_names = [name for name in _UV_csv_names for _ in range(3)]

    # Search for corresponding .csv files.
    scene_pairs = {}
    print("Matching scene pairs...")
    for UV_csv_name in UV_csv_names:
        # Extract the base file name (e.g., 'imgA.bmp')
        bmp_name = os.path.basename(UV_csv_name)
        scene_number = bmp_name.split('-')[2].replace(".csv","")
        # Find the corresponding file in csv_name based on the common pattern
        # for n in fringe_names:
        #     print(scene_number, os.path.basename(n).split("-")[2])
        selected_fringe_names = [n for n in fringe_names if scene_number in os.path.basename(n).split("-")[2]]

        # # If there are matches, add them to the result
        if len(selected_fringe_names)>0:
            for n in selected_fringe_names:
                scene_pairs[n] = UV_csv_name

    # Create images
    for k, v in tqdm(scene_pairs.items()):
        fringe_name = k
        corresponding_UV_csv_name = v
        grayscale_img = imread(fringe_name)
        lines = read_UV_csv(corresponding_UV_csv_name)
        save_name = fringe_name.split("/")[-1].replace(".bmp","_UV.bmp")
        idx = int(fringe_name.split("-")[-1].split(".")[0].replace("img",""))

        # grayscale_img = np.stack((grayscale_img,)*1, axis=-1)
        _, centerline_img = line_augmentation(grayscale_img, lines, k=idx) # gth
        # save UV only
        # print(os.path.join(save_path_UV, f"{save_name}_UV.bmp"))
        imsave(os.path.join(save_path_UV, save_name), centerline_img)

    print("fringe_names:",len(fringe_names))
    print("UV_csv_names:",len(UV_csv_names))
    print(f"Total: {len(scene_pairs)} pairs of image(s).")
    print(f"{len(fringe_names)-len(scene_pairs)} fringe image(s) are filtered.")
    print("Done")

if __name__ == '__main__':

    # 把所有 00,01,02 預先加工成有線心的樣子
    parser = argparse.ArgumentParser()
    """
    data_root: path of images.
    save_path_UV: path to save centerline position images
    """
    parser.add_argument('--data_root', type=str, default= "/home/pywu/Downloads/zhong/dataset/teeth_qisda/imgs/0727-0933/")
    parser.add_argument('--UV_root', type=str, default= "/home/pywu/Downloads/zhong/dataset/teeth_qisda/supplements/0727-0933/UV/") # "/home/pywu/Downloads/zhong/dataset/teeth_qisda/0727-0933_UV"
    parser.add_argument('--save_path_UV', type=str, default= "/home/pywu/Downloads/zhong/dataset/teeth_qisda/supplements/0727-0933/UV_only") # "/home/pywu/Downloads/zhong/dataset/teeth_qisda/0727-0933_UV"
    
    args = parser.parse_args()

    verbose = True
    data_root = args.data_root
    UV_root = args.UV_root
    save_path_UV = args.save_path_UV

    os.makedirs(save_path_UV, exist_ok=True)

    make_data(data_root, UV_root, save_path_UV)
    #print(sorted(glob.glob(os.path.join(data_root, "*.bmp"), recursive=True)))
    #print(len(glob.glob(os.path.join(data_root, "*.bmp"))))