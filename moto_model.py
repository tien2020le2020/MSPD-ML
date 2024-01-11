from moto_utils import *
set_seed(seed)

import numpy as np
from sklearn.metrics import accuracy_score
import pickle
from tqdm import tqdm

from sklearn import metrics
from catboost import Pool, CatBoostClassifier

scale_distance = MAX_V # 1024

def get_atomic_features(row, nb_points, obj_col, is_target=True,
                 gammas_0 = [0.7, 0.5], gammas_1 = [0.7, 0.5], nb_diffuses = 2,
                 normalized = True):
    netIdx = row[f"netIdx"]
    points = [(row[f"x{i}"],row[f"y{i}"]) for i in range(nb_points)]

    distances = scipy.spatial.distance.cdist(points, points, 'euclidean') / scale_distance

    df_features = pd.DataFrame()
    df_features['netIdx'] = [netIdx] * nb_points
    df_features['nb_points'] = [nb_points] * nb_points
    df_features['obj_col'] = [obj_col] * nb_points
    df_features['point'] = list(range(nb_points))
    df_features['net_id'] = [f"{netIdx}_{nb_points}_{obj_col}"] * nb_points
    df_features['coord'] = points

    if is_target:
        target_masks = [int(row[f"{i}"]) for i in range(nb_points)]
        df_features['mask'] = target_masks

    for col in ["run", "run2"]:
        if col in row:
            df_features[col] = [row[col]] * nb_points

    # distance
    df_features['gamma_distance'] = distances[0,:]

    # Origin node: diffuse the score
    for k, gamma in enumerate(gammas_0):
        root_scores = np.zeros(nb_points)
        root_scores[0] = 1

        gamma_scores = (root_scores * (gamma ** distances)).sum(axis=1) # first sum
        df_features[f'gamma_0_{k}_0'] = gamma_scores

        for i in range(1, nb_diffuses+1):
            gamma_scores = (gamma_scores * (gamma ** distances)).mean(axis=1)
            df_features[f'gamma_0_{k}_{i}'] = gamma_scores

    # network nodes: diffuse the distance
    for k, gamma in enumerate(gammas_1):
        root_scores = np.ones(nb_points)
        gamma_scores = (root_scores * (gamma ** distances)).mean(axis=1)
        df_features[f'gamma_1_{k}_0'] = gamma_scores

        for i in range(1, nb_diffuses+1):
            gamma_scores = (gamma_scores * (gamma ** distances)).mean(axis=1)
            df_features[f'gamma_1_{k}_{i}'] = gamma_scores

    if normalized:
        for c in df_features.columns:
            if "gamma" in c:
                m2 = df_features[c].max()
                m1 = df_features[c].min()
                df_features[c] = (df_features[c] - m1)/(m2-m1)

    return df_features

def get_full_features(n, objectiveN, netInputDf, is_target=True, verbose=True):
    df_features = []
    nb_points = n
    obj_col = f"obj{objectiveN}" if len(str(objectiveN)) == 1 else objectiveN

    df_net = netInputDf.copy()
    # print("Before FE:", df_net.shape)

    nb_images = df_net.shape[0]
    if verbose:
        for row_index in tqdm(range(nb_images)):
            row = df_net.iloc[row_index]
            df_features.append(get_atomic_features(row, nb_points, obj_col, is_target=is_target))
    else:
        for row_index in range(nb_images):
            row = df_net.iloc[row_index]
            df_features.append(get_atomic_features(row, nb_points, obj_col, is_target=is_target))

    df_features = pd.concat(df_features).reset_index(drop=True)

    agg_raw_feature_cols = [c for c in df_features.columns if "gamma" in c]

    df_agg = df_features.groupby("net_id")[agg_raw_feature_cols].agg(["mean", "std", "skew"])

    flat_cols = []
    for i in df_agg.columns:
        flat_cols.append(i[0]+'_'+i[1])
    df_agg.columns = flat_cols

    df_agg = df_agg.reset_index()

    df_features2 = pd.merge(df_features, df_agg, on="net_id")
    feature_cols2 = ["obj_col", "nb_points"]
    feature_cols2 += [c for c in df_features2.columns if "gamma" in c]

    # print("After FE:", df_features2.shape)
    return feature_cols2, df_features2

def train_model(df_net, df_features, nb_points, obj_col,
                nb_folds=5, fold=0, models_dir="models"):


    # Folds split
    df_folds = df_net[["netIdx"]].drop_duplicates().sort_values("netIdx").reset_index(drop=True)
    df_folds["fold"] = df_folds.index % nb_folds
    df_folds["fold"].value_counts()

    df_folds = pd.merge(df_folds, df_features, on="netIdx")
    df_folds.tail()

    # Sampling
    key_cols = "netIdx	net_id	run	run2	point".split()
    target_col = "mask"


    def downsample(df_features, ratio=2):
        df_full = df_features[(df_features["point"] != 0)]
        df_pos = df_full[df_full[target_col] == 1]
        df_neg = df_full[df_full[target_col] == 0]
        nb_pos = df_pos.shape[0]
        nb_neg = df_neg.shape[0]
        df_neg = df_neg.sample(n = int(nb_pos*ratio))
        df_samples = pd.concat([df_pos, df_neg])
        df_samples = df_samples.sort_values(key_cols).reset_index(drop=True)
        return df_samples

    df_samples = downsample(df_folds, ratio=2)

    # Modelling

    feature_cols = [c for c in df_samples.columns if "gamma" in c]
    print("downsample:", df_folds.shape, df_samples.shape)
    print(feature_cols)

    if fold >= 0:
        df_val = df_samples[(df_samples["fold"] == fold)].reset_index(drop=True)
        df_train = df_samples[(df_samples["fold"] != fold)].reset_index(drop=True)
    else:
        df_val = df_samples.reset_index(drop=True)
        df_train = df_samples.reset_index(drop=True)

    print(df_train.shape, df_val.shape)

    run_feature_cols = feature_cols[:]
    X_val = df_val[run_feature_cols]
    X_train = df_train[run_feature_cols]

    y_val = df_val[target_col]
    y_train = df_train[target_col]

    print(X_train.shape, X_val.shape)

    cat_features = []
    train_dataset = Pool(data=X_train,
                         label=y_train,
                         cat_features=cat_features)
    val_dataset = Pool(data=X_train,
                         label=y_train,
                         cat_features=cat_features)

    model = CatBoostClassifier(iterations=20000)
    # Fit model with `use_best_model=True`
    model.fit(train_dataset,
              use_best_model=True,
              eval_set=val_dataset,
              verbose=1000)

    # Validation - ACC
    y_pred = model.predict(X_val).flatten()
    accuracy = accuracy_score(y_val, y_pred)
    print("Accuracy:", accuracy)

    pd.Series(y_val).value_counts(normalize=True)

    # Validaton - AUC
    y_pred = model.predict_proba(X_val)[:,1]
    # pd.Series(y_pred).hist(bins=30)

    fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred, pos_label=1)
    auc = round(metrics.auc(fpr, tpr), 3)
    print("auc:", auc)

    model_filename = f"models/catboost_{nb_points}_{obj_col}_fold{fold}.cbm"
    print(f"Saving model to {model_filename} ...")
    model.save_model(model_filename)

    return model_filename
