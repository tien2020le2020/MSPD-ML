#!/usr/bin/env python
# coding: utf-8

from moto_utils import *
from moto_model import *

from datetime import datetime
from pathlib import Path

models_dir = "models"
models_path = Path(models_path)
models_path.mkdir(parents=True, exist_ok=True)

def print_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("", current_time)


set_seed(seed)

os.system('python --version')

buffer_dir = "workspace"
def get_df_aug(nb_points, obj_col):
    df_net = pd.read_csv(f"{buffer_dir}/df_aug2_{nb_points}_{obj_col}.csv")
    return df_net



list_supported_nb_points = [10, 15, 25, 30, 40, 45, 50]
fold = -1

for nb_points in list_supported_nb_points:
    print(f"Processing {nb_points}")
    for obj_col in list_objs:
        print_time()
        print("-"*20, "Modelling", "-"*20)
        print("Meta:", nb_points, obj_col)

        df_net = get_df_aug(nb_points, obj_col)

        print("Features creating ...")
        feature_cols, df_features = get_full_features(nb_points, obj_col, df_net, is_target=True)
        print("feature_cols:", feature_cols)

        print("Training ...")
        model_filename = train_model(df_net, df_features, nb_points, obj_col,
                nb_folds=5, fold=fold, models_dir=models_dir)
        print(nb_points, obj_col, "=>", model_filename)
        print("-"*20, "-----", "-"*20)
        print_time()

print("OK")
