import gzip
import json
from itertools import islice
from pathlib import Path

import requests
from bs4 import BeautifulSoup


def gen_task_indices(index):
    index = min(index, 185)

    dest_path = "data/sitemap"
    if not Path(dest_path).exists():
        Path(dest_path).mkdir(parents=True)

    # Download xml file if ot exists
    filename = f"{dest_path}/recipe_list_{index}.xml"
    if not Path(filename).exists():
        url = f"https://www.xiachufang.com/sitemap/recipe_list_{index}.xml.gz"
        response = requests.get(url).content
        with open(f"{dest_path}/recipe_list_{index}.xml", "wb") as f:
            f.write(gzip.decompress(response))
        print("===> Downloaded xml file.")

    # Read indices
    print("===> Generating task indices...")
    indices = []
    with open(filename) as f:
        for line in f:
            chef_soup = BeautifulSoup(line, "xml")
            recipe_url = chef_soup.find("loc")
            if recipe_url:
                indices.append(int(recipe_url.text.split('/')[-2]))
    print("===> [Done]")
    return indices


def json_saver(filename, content):
    """Save with encoding utf-8 for readability"""

    if not Path(filename).parent.exists():
        Path(filename).parent.mkdir()
    with open(filename, 'w', encoding='utf8') as json_file:
        json.dump(content, json_file, ensure_ascii=False)


def do_saver(idx, chef_lists):
    json_saver(gen_savename(idx - len(chef_lists), idx), chef_lists)
    chef_lists.clear()
    return chef_lists


def split_task(it, task_nums=4):
    it = iter(it)
    return iter(lambda: tuple(islice(it, task_nums)), ())


def gen_savename(lhs, rhs, prefix="data/starred/recipe"):
    name = prefix + f'_{lhs:08}_{rhs:08}.json'
    return name


def is_chinese(query):
    ch = list(query)[0]
    return u'\u4e00' <= ch <= u'\u9fa5'
