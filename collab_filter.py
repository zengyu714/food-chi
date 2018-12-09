"""
Ref: https://github.com/NicolasHug/Surprise/blob/711fb80748/examples/top_n_recommendations.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from surprise import Dataset,Reader, SVD, NMF, dump
from surprise.accuracy import rmse
from surprise.model_selection import GridSearchCV, KFold

MIN_RECIPES_PER_USER = 5
MIN_USER_PER_RECIPES = 50
KFOLD_NUM = 3
THRESHOLD = 4.95
TOP_K = 5


def top_k(predictions, top_num=3):
    top_recs = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_recs[uid].append((iid, est))

    for uid, user_ratings in top_recs.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_recs[uid] = user_ratings[:top_num]

    return top_recs


def dict2index(inverse=False):
    dictionary = Path("data/preprocessed/dictionary.json")
    recipe_index = {}
    for idx, recipe in enumerate(json.load(dictionary.open(encoding="utf-8")).keys()):
        recipe_index[recipe] = idx
    if inverse:
        recipe_index = {v: k for k, v in recipe_index.items()}
    return recipe_index


def gen_user_rate(user, recipes, dictionary):
    cnt = Counter()
    for recipe in recipes:
        cnt[recipe] += 1
    rows = []
    for k, v in cnt.items():
        # cast from [1, 6] to [5, 10]
        v += 4
        v = min(v, 10)
        try:
            rows.append([user, dictionary[k], v])
        except KeyError:
            continue
    return rows


def gen_dataframe():
    """
    Data statistics: `np.histogram(df.rating, bins=28, range=(5, 33))`
    - (Before) rating from [1, 18]
    [2335114, 149570, 23281, 5907, 1910, 847, 435, 199, 111, 86, 19, 13, 25, 13, 10, 3, 11, 0, 10, 1, 0, 0, 6, 0, 0, 0, 1]
    - (After) rating from [5, 10]


    """
    savepath = Path("data/preprocessed/collaborative_df.pkl")
    if savepath.exists():
        df = pd.read_pickle(str(savepath))
    else:
        dictionary = dict2index()
        rows = []
        user_id = 0
        for json_file in Path("data/preprocessed/").glob("mapped_starred*"):
            print(f"Processing {json_file}")
            for recipes in json.load(json_file.open(encoding="utf-8")).values():
                row = gen_user_rate(user_id, recipes, dictionary)
                if len(row) > MIN_RECIPES_PER_USER:
                    rows.extend(row)
                    user_id += 1

        user_id, recipe_id, rating = list(zip(*rows))
        rating_dict = {"user_id": user_id, "recipe_name": recipe_id, "rating": rating}
        df = pd.DataFrame(rating_dict)

        # filter out most least favorite recipes
        recipe_count = df.groupby(by=["recipe_name"]).count()["user_id"]
        unpopular_ind = recipe_count[recipe_count < MIN_USER_PER_RECIPES].index.values
        df = df.loc[~df["recipe_name"].isin(unpopular_ind)]
        df.to_pickle(str(savepath))
    return df


def load_dataset():
    df = gen_dataframe()
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(df[['user_id', 'recipe_name', 'rating']], reader)
    return data


def load_trained_pred_algo(model_path):
    predictions, algo = dump.load(model_path)
    df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
    return df.rename(index=str, columns={"uid": "user_id", "iid": "item_id", "rui": "rating", "est": "estimation"}).drop("details", axis=1), predictions, algo


def train_helper(algo, savename, trainset_cv, testset_cv):
    algo.fit(trainset_cv)
    print(f"{savename} on trainset")
    predictions = algo.test(trainset_cv.build_testset())
    rmse(predictions, verbose=True)
    print(f"{savename} on testset")
    predictions = algo.test(testset_cv)
    rmse(predictions, verbose=True)
    dump.dump(f"models/dump_{savename}", predictions, algo)


def train():
    data = load_dataset()
    # algo_svd = SVD()
    algo_nmf = NMF()

    print('CV procedure:')
    kf = KFold(n_splits=KFOLD_NUM)
    for i, (trainset_cv, testset_cv) in enumerate(kf.split(data), start=1):
        print('fold number', i)
        # train_and_save(algo_svd, "SVD", trainset_cv, testset_cv)
        train_helper(algo_nmf, "NMF", trainset_cv, testset_cv)


def estimate():
    predictions_svd, algo_svd = dump.load('models/dump_SVD')
    precisions, recalls = precision_recall_at_k(predictions_svd, k=TOP_K, threshold=THRESHOLD)

    df_svd = pd.DataFrame(predictions_svd, columns=['uid', 'iid', 'rui', 'est', 'details'])
    df_svd['err'] = abs(df_svd.est - df_svd.rui)

    with open("estimation.txt", "w+") as f:
        f.write(f"SVD\n{df_svd.head()}\n")
        # Precision and recall can then be averaged over all users
        f.write(f"Precision: {sum(prec for prec in precisions.values()) / len(precisions)}\n")
        f.write(f"Recall: {sum(rec for rec in recalls.values()) / len(recalls)}\n")


def cal_recall(predictions, err_threshold=0.05):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    recalls = {}
    for uid, user_ratings in user_est_true.items():
        ratings = np.array(user_ratings)
        tp = ratings[:, 1]
        tp_num = tp.shape[0]

        fn = ratings[np.where((ratings[:, 1] - ratings[:, 0]) > err_threshold)][:, 0]
        fn_num = fn.shape[0]
        recalls[uid] = tp_num / (tp_num + fn_num)
    return recalls

def precision_recall_at_k(predictions, k, threshold):
    """Return precision and recall at k metrics for each user."""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls


if __name__ == "__main__":
    train()
    estimate()
