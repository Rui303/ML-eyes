#!/usr/bin/python3
import os
from PIL import Image

def resize_images(directory, width, height):
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.png'):
                filepath = os.path.join(foldername, filename)
                with Image.open(filepath) as img:
                    img_resized = img.resize((width, height))
                    img_resized.save(filepath)
                    print(f"Resized and saved: {filepath}")

# 使用示例
dir_path = "/home/nvidia/fianl-project/data2"
resize_images(dir_path, 320, 180)
