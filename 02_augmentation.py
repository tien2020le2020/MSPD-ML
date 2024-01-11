#!/usr/bin/env python
# coding: utf-8

from moto_utils import *
os.system('python --version')

set_seed(seed)

buffer_dir = "workspace"
def get_df_net(nb_points, obj_col):
    df_net = pd.read_csv(f"{buffer_dir}/df_net_{nb_points}_{obj_col}.csv")
    return df_net

for nb_points in list_nb_points:
    print(f"Processing {nb_points}")
    for obj_col in tqdm(list_objs):

        df_net = get_df_net(nb_points, obj_col)

        df_aug = aug_run(df_net, nb_points=nb_points, max_move=0.05, nb_runs=5)
        df_aug2 = aug_run2(df_aug, nb_points, nb_runs=10)

        output_filename = f"{buffer_dir}/df_aug2_{nb_points}_{obj_col}.csv"
        df_aug2.to_csv(output_filename, index=None)

        print("Processed:", nb_points, obj_col, "shape:", df_net.shape, df_aug2.shape)

print("OK")
