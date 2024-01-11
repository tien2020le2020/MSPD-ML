import scipy
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from moto_utils import *
from moto_model import *

def get_all_models():
    dict_models = {}

    fold = -1
    counter = 0
    for nb_points in list_nb_points:
        dict_models[nb_points] = {}

        for obj_col in list_objs:
            model_filename = f"models/catboost_{nb_points}_{obj_col}_fold{fold}.cbm"

            try:
                model = CatBoostClassifier()
                model.load_model(model_filename)
                dict_models[nb_points][obj_col] = model
                counter += 1
            except:
                print(f"Unable to load {model_filename} !")

    print(f"Loaded {counter} models")
    return dict_models

dict_models = get_all_models()
debug = False # len(dict_models) < 7

# Get candiates (source node, first node, and second node)
def calculate_angle(source, p1, p2):
    ba = p1 - source
    bc = p2 - source

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return int(abs(np.degrees(angle)))

def get_inference_points(df_features, nb_points, obj_col,
                            nb_top=10, min_prob=0.025,
                            min_angle_threshold = 75,
                            max_nb_returns=3):
    feature_cols = [c for c in df_features.columns if "gamma" in c]
    model = dict_models[nb_points][obj_col]
    # Get the source node
    source_node = np.array(df_features[df_features["point"] == 0].iloc[0]["coord"])

    # Pred
    X_test = df_features[feature_cols]
    y_pred = model.predict_proba(X_test)[:,1]
    df_features["pred"] = y_pred

    view_cols = [c for c in df_features.columns if ("gamma" not in c) or ("gamma_distance" == c)]
    df_post = df_features[df_features["point"] != 0][view_cols].copy()

    # Keep top prob with at least 2.5%
    df_post = df_post[(df_post["pred"] >= min_prob)].reset_index(drop=True)
    df_post['rank'] = df_post.groupby('net_id')['pred'].rank("dense", ascending=False)
    df_post['rank'] = df_post['rank'].astype(int)

    df_post = df_post[(df_post["rank"] <= nb_top)].sort_values("rank").reset_index(drop=True)
    if df_post.shape[0] == 0: # In case no other nodes, return empty list
        return []

    df_cur = df_post

    first_node = np.array(df_cur.iloc[0]["coord"])
    first_point = df_cur.iloc[0]["point"]

    target_points = [first_point]

    # Second node if possible
    nodes = [np.array(e) for e in df_cur["coord"].values]

    second_candidate_index = 1000
    second_candidate_node = None
    second_candidate_point = -1

    for i, node in enumerate(nodes):
        if i == 0: # first target node
            continue

        if calculate_angle(source_node, first_node, node) >= min_angle_threshold:
            second_candidate_index = i
            second_candidate_node = node
            second_candidate_point = df_cur.iloc[i]["point"]
            break

    if second_candidate_point > 0:
        target_points = [first_point, second_candidate_point]

        third_candidate_index = 1000
        third_candidate_node = None
        third_candidate_point = -1

        for i, node in enumerate(nodes):
            if i <= second_candidate_index: # already removed
                continue

            if (calculate_angle(source_node, first_node, node) >= min_angle_threshold) and \
                (calculate_angle(source_node, second_candidate_node, node) >= min_angle_threshold):
                third_candidate_index = i
                third_candidate_node = node
                third_candidate_point = df_cur.iloc[i]["point"]
                break

        if third_candidate_point > 0:
            target_points = [first_point, second_candidate_point, third_candidate_point]

    return target_points

counter = 0
def run_inference(N, objectiveN, inputDf):
    nb_points = N
    obj_col = objectiveN if len(str(objectiveN)) == 4 else f"obj{objectiveN}"
    df_net = inputDf.copy()

    feature_cols, df_features = get_full_features(nb_points, obj_col, df_net,
                                    is_target=False, verbose=False)
    points = get_inference_points(df_features, nb_points, obj_col)

    if debug:
        print(f"run_inference ({counter}):", nb_points, obj_col, "=>", points)

    return points
