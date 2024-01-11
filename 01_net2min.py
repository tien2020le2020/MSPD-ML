#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

input_dir = "/mnt/testcases/testcases/"
workspace_dir = "workspace"


# Create the directory
workspace_path = Path(workspace_dir)
workspace_path.mkdir(parents=True, exist_ok=True)

dict_nets = {}

for n in tqdm([10, 15, 25, 30, 40, 45, 50]):
    print(f"Processing {n} ...")
    dict_nets[n] = {}

    for obj_col in ["obj1", "obj2", "obj3"]:
        sorted_cols = [c for c in ["obj1", "obj2", "obj3"] if c != obj_col]
#         print("Objective:", obj_col, sorted_cols)

        dataObjDf = pd.read_csv(f"{input_dir}/data_obj_stt_%d.csv.gz" %(n), compression="gzip")
        inputDf = pd.read_csv(f"{input_dir}/input_stt_%d.csv.gz" %(n), compression="gzip")
        sourceDf = pd.read_csv(f"{input_dir}/sources_stt_%d.csv.gz" %(n), compression="gzip")

        df_min = dataObjDf.groupby(["netIdx"])[obj_col].min().to_frame(obj_col).reset_index()
        df_min = pd.merge(df_min, dataObjDf, on=["netIdx", obj_col])
        df_min = df_min.sort_values(["netIdx", obj_col] + sorted_cols).reset_index(drop=True)
        df_min = df_min.drop_duplicates(subset=["netIdx"]).reset_index(drop=True)

        nets_min = pd.merge(df_min, sourceDf, on=["sourceIdx"])
        nets_min = pd.merge(nets_min, inputDf, on=["netIdx"])

        dict_nets[n][obj_col] = nets_min
        """
        for i in range(n):
            # Origin node
            if i == 0:
                image[yi,xi, 0] = image_value
            # Network nodes
            image[yi,xi, 1] = image_value
            # Target (or wanted / or to be found) nodes
            if int(row[f"{i}"]) == 1:
                image[yi,xi, 2] = image_value
        """
        output_filename = f"{workspace_dir}/df_net_{n}_{obj_col}.csv"
        nets_min.to_csv(output_filename, index=None)
print("OK")
