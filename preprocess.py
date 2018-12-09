"""
Pre-process user data
1. Convert json file to csv (Filter out empty items)
2. Word segmentation
"""

import json
from pathlib import Path

import docx
from joblib import Parallel, delayed
from similarity.jarowinkler import JaroWinkler
from tqdm import tqdm

from utils.tools import json_saver


def filter_empty():
    root_dir = Path("data/starred")
    sub_dirs = [d for d in root_dir.iterdir() if d.is_dir()]

    total_num = 0
    for sub_dir in sub_dirs:
        giant = {}
        json_names = sub_dir.iterdir()
        for json_name in json_names:
            curr_json = json.load(json_name.open(encoding="utf-8"))
            curr_json = {k: v for k, v in curr_json.items() if k != "NA"}
            giant.update(curr_json)

        savename = f"data/preprocessed/{sub_dir.stem}.json"
        total_num += len(giant)
        print(sub_dir, len(giant), total_num)
        json_saver(savename, giant)


def construct_dictionary():
    dictionary = {}
    doc = docx.Document("data/preprocessed/translation.docx")
    total_size = 0
    for table in doc.tables:
        cur_size = 0
        for i in tqdm(range(len(table.rows))):
            head, zh, en = [table.cell(i, j).text for j in range(3)]
            zh, en = zh.strip(), en.strip()
            # print(head, zh, en)
            if not head:
                dictionary[zh] = en
                cur_size += 1
        total_size += cur_size
    print(f"Dictionary size: {total_size}")
    json_saver("data/preprocessed/dictionary.json", dictionary)


def map_to_dictionary(sim_tool, dirty, dictionary):
    results = {word: sim_tool.similarity(dirty, word) for word in dictionary}
    sorted_results = sorted(results.items(), key=lambda d: d[1], reverse=True)
    best = sorted_results[0]
    return best[0] if best[1] > .5 else ""


def map_recipes_worker(json_file):
    # Load similarity tools
    dictionary = json.load(open("data/preprocessed/dictionary.json", encoding="utf-8"))
    dictionary_words = dictionary.keys()

    jarowinkler = JaroWinkler()
    print(f"Processing {json_file}")
    curr_json = json.load(json_file.open(encoding="utf-8"))
    processed_json = {}
    for user, recipes in tqdm(curr_json.items()):
        processed_json[user] = []
        for recipe in recipes:
            mapped = map_to_dictionary(jarowinkler, recipe, dictionary_words)
            if mapped:
                processed_json[user].append(mapped)
    processed_json = {k: v for k, v in processed_json.items() if len(v)}
    json_saver(f"data/preprocessed/mapped_{json_file.stem}.json", processed_json)


def map_recipes_master():
    json_files = Path("data/preprocessed/").glob("starred*")
    Parallel(n_jobs=-1)(delayed(map_recipes_worker)(json_file) for json_file in json_files)


if __name__ == "__main__":
    # filter_empty()
    # construct_dictionary()
    map_recipes_master()
