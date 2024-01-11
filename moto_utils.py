#!/usr/bin/env python
# coding: utf-8

import os


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import scipy
import pickle
import itertools
import glob, os
from sklearn.model_selection import StratifiedKFold

import random
import math

seed = 2023

MIN_V, MAX_V = 0, 1024
image_size = 16
image_source_value = 3
image_node_value = 1
image_target_value = 2

list_nb_points = [10, 15, 25, 30, 40, 45, 50]
list_objs = ['obj1', 'obj2', 'obj3']

def set_seed(seed=seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
def aug_noise(df_net, nb_points, max_move=0.05):
    df_aug = df_net.copy()
    nb_rows = df_net.shape[0]
    max_move_v = max_move*MAX_V
        
    for i in range(nb_points):
        for c in [f"x{i}", f"y{i}"]:
            series_m = (pd.Series(np.random.random_sample(nb_rows))-0.5)*max_move_v
            df_aug[c] = df_aug[c] + series_m
            df_aug[c] = df_aug[c].astype(int)
            df_aug[c] = df_aug[c].clip(MIN_V, MAX_V)
    
    return df_aug

def aug_flip(df_net, nb_points, flip_type=0):
    df_aug = df_net.copy()
    
    if flip_type == 0:
        return df_aug
    
    if flip_type == 1:
        for i in range(nb_points):
            for c in [f"x{i}"]:
                df_aug[c] = MAX_V - df_aug[c]

    if flip_type == 2:
        for i in range(nb_points):
            for c in [f"y{i}"]:
                df_aug[c] = MAX_V - df_aug[c]
                
    if flip_type == 3:
        for i in range(nb_points):
            for c in [f"x{i}", f"y{i}"]:
                df_aug[c] = MAX_V - df_aug[c]
    
    return df_aug

def aug_swap(df_net, nb_points, swap_type=0):
    df_aug = df_net.copy()
    
    if swap_type == 0:
        return df_aug
    
    if swap_type == 1:
        for i in range(nb_points):
            c1 = f"x{i}"
            c2 = f"y{i}"
            s1 = df_aug[c1].copy()
            df_aug[c1] = df_aug[c2]
            df_aug[c2] = s1
    
    return df_aug

def aug_rotate(df_net, nb_points, rotation_type=0):
    df_aug = df_net.copy()
    
    if rotation_type == 0:
        return df_aug
    
    if rotation_type in [1, 90]:
        df_aug = aug_swap(df_aug, nb_points, swap_type=1)
        df_aug = aug_flip(df_aug, nb_points, flip_type=1)

    if rotation_type in [2, 180]:
        df_aug = aug_flip(df_net, nb_points, flip_type=3)
        
    if rotation_type in [3, -90, 270]:
        df_aug = aug_swap(df_aug, nb_points, swap_type=1)
        df_aug = aug_flip(df_aug, nb_points, flip_type=2)
    
    return df_aug

def aug_run(df_net, nb_points, max_move=0.05, nb_runs=10):
    df_aug = df_net.copy()
    df_aug["run"] = 0
    
    df_augs = [df_aug]
    for i in range(1, nb_runs+1):
        df_aug = aug_noise(df_net, nb_points)
        df_aug["run"] = i
        
        df_aug = aug_flip(df_aug, nb_points, flip_type=random.choice([0,1,2,3]))
        df_aug = aug_swap(df_aug, nb_points, swap_type=random.choice([0,1]))
        df_aug = aug_rotate(df_aug, nb_points, rotation_type=random.choice([0,1,2,3]))
        
        df_augs.append(df_aug)
    df_augs = pd.concat(df_augs).reset_index(drop=True)
    
    return df_augs

def get_image(row, nb_points):
    image = np.zeros((image_size,image_size), dtype="uint8")
    for i in range(nb_points):
        xi = row[f"x{i}"]
        yi = row[f"y{i}"]
        xi = min(int(xi*image_size/MAX_V), image_size-1)
        yi = min(int(yi*image_size/MAX_V), image_size-1)
        
        # Origin node
        if i == 0:
            image[yi,xi] = max(image_source_value, image[yi,xi])
        # Network nodes
        image[yi,xi] = max(image_node_value, image[yi,xi])

    target_masks = [int(row[f"{i}"]) for i in range(nb_points)]
    nb_masks = sum(target_masks)
    return image, nb_masks

def get_image_with_targets(row, nb_points):
    image = np.zeros((image_size,image_size), dtype="uint8")
    for i in range(nb_points):
        xi = row[f"x{i}"]
        yi = row[f"y{i}"]
        xi = min(int(xi*image_size/MAX_V), image_size-1)
        yi = min(int(yi*image_size/MAX_V), image_size-1)
        
        # Origin node
        if i == 0:
            image[yi,xi] = max(image_source_value, image[yi,xi])
        # Network nodes
        image[yi,xi] = max(image_node_value, image[yi,xi])
        # Target (or wanted / or to be found) nodes
        if int(row[f"{i}"]) == 1:
            image[yi,xi] = max(image_target_value, image[yi,xi])
                
    target_masks = [int(row[f"{i}"]) for i in range(nb_points)]
    nb_masks = sum(target_masks)
    return image, nb_masks

def get_image_only(row, nb_points):
    image = np.zeros((image_size,image_size), dtype="uint8")
    for i in range(nb_points):
        xi = row[f"x{i}"]
        yi = row[f"y{i}"]
        xi = min(int(xi*image_size/MAX_V), image_size-1)
        yi = min(int(yi*image_size/MAX_V), image_size-1)
        
        # Origin node
        if i == 0:
            image[yi,xi] = max(image_source_value, image[yi,xi])
        # Network nodes
        image[yi,xi] = max(image_node_value, image[yi,xi])                
    return image

def rotate_point(x, y, x0, y0, alpha):
    # Convert angle to radians
    alpha = math.radians(alpha)

    # Calculate the rotated point
    new_x = (x - x0) * math.cos(alpha) - (y - y0) * math.sin(alpha) + x0
    new_y = (x - x0) * math.sin(alpha) + (y - y0) * math.cos(alpha) + y0

    return new_x, new_y

def rotate_df_net_around_point(df_net, nb_points, x0, y0, alpha):
    df_aug = df_net.copy()
    
    # Convert angle to radians
    alpha = math.radians(alpha)
    sin_alpha, cos_alpha = math.sin(alpha), math.cos(alpha)
    
    cc_cols = []
    
    for i in range(nb_points):
        # Calculate the rotated point
        cx, cy = f"x{i}", f"y{i}"
        new_x = (df_aug[cx] - x0) * cos_alpha - (df_aug[cy] - y0) * sin_alpha + x0
        new_y = (df_aug[cx] - x0) * sin_alpha + (df_aug[cy] - y0) * cos_alpha + y0
        
        df_aug[cx] = new_x
        df_aug[cy] = new_y

        cc_cols += [cx, cy]
        
    min_v, max_v = df_aug[cc_cols].values.min(), df_aug[cc_cols].values.max()        
    for col in cc_cols:
        df_aug[col] = (df_aug[col]-min_v)*(MAX_V/(max_v-min_v))
        df_aug[col] = df_aug[col].astype(int).clip(MIN_V, MAX_V)        

    return df_aug

def rotate_df_net(df_net, nb_points, max_shift=0.1, max_angle=20):
    x0, y0 = MAX_V//2, MAX_V//2
    
    max_r = max_shift * MAX_V    
    x0 = x0 + (np.random.rand()-0.5) * max_r
    y0 = y0 + (np.random.rand()-0.5) * max_r
    
    angle = (np.random.rand()-0.5)*max_angle
    
    df_aug = rotate_df_net_around_point(df_net, nb_points, x0, y0, angle)
    return df_aug

def aug_run2(df_net, nb_points, max_shift=0.1, max_angle=20, nb_runs=10):
    df_aug = df_net.copy()
    df_aug["run2"] = 0
    
    df_augs = [df_aug]
    for i in range(1, nb_runs+1):
        df_aug = rotate_df_net(df_net, nb_points, max_shift=0.1, max_angle=20)
        df_aug["run2"] = i
        
        df_augs.append(df_aug)
    df_augs = pd.concat(df_augs).reset_index(drop=True)
    
    return df_augs

