"""
First download xml.gz compressed file from xiachufang
E.g.,
https://www.xiachufang.com/sitemap/recipe_list_0.xml.gz
https://www.xiachufang.com/sitemap/recipe_list_1.xml.gz
...
https://www.xiachufang.com/sitemap/recipe_list_185.xml.gz

Each xml contains ~49500 lists
"""

import random
import time
from collections import defaultdict

import click
from bs4 import BeautifulSoup
from joblib import Parallel, delayed

from utils.mail_msg import send_msg
from utils.tools import split_task, do_saver, gen_task_indices
from utils.proxifier import request_proxifier


def data_parser(tasks, frequency):
    """Parse xml and get recipe list from website"""

    lower, upper = int(min(tasks)), int(max(tasks))
    chef_lists = defaultdict(list)
    alert_count = 1

    for idx, recipe_index in enumerate(tasks, start=1):
        recipe_index = int(recipe_index)
        print(f">>>> Processing {recipe_index} / {upper}...")
        time.sleep(random.randint(1, 3))

        url = f"http://www.xiachufang.com/recipe_list/{recipe_index}/"
        try:
            response = request_proxifier(url)
        except:
            print("===> Fail to GET site")
            continue

        if not response:
            # Something wrong like ConnectionError / InvalidURL
            print(f"===> Skip {recipe_index} since {response.reason}")
            if len(chef_lists) > frequency / 5:
                print(f"===> Saving {recipe_index} / {upper} [{recipe_index - lower}/{upper - lower}]...")
                chef_lists = do_saver(recipe_index, chef_lists)
            if response.reason == "Too Many Requests" or response.status_code == 429:
                alert_count += 1
                if alert_count > 20:
                    print(f"===> Sending email alerts")
                    send_msg(f"Oops, too many requests response")
                    break
        else:
            chef_soup = BeautifulSoup(response.content, "html5lib")
            chef_name = chef_soup.find("a", class_="avatar-link")
            recipes_all = chef_soup.find_all("p", class_="name")

            try:
                chef_name = chef_name.text.strip()
                recipes_all = [ele.a.contents[0] for ele in recipes_all]
            except:
                chef_name = "NA"
                recipes_all = ["empty"]

            chef_lists[chef_name].extend(recipes_all)

            # Monitor - Console
            if (idx % 10) == 0:
                print(f"\n{chef_name}: {chef_lists[chef_name]}")

            # Save data
            if len(chef_lists) and idx % frequency == 0:
                print(f"===> Saving {recipe_index} / {upper} [{recipe_index - lower}/{upper - lower}]...")
                chef_lists = do_saver(recipe_index, chef_lists)


@click.command()
@click.option('--index', default=0, prompt='Task begins with', help='Represents the index of xml.')
@click.option('--cores', default=8, help='Nums of cpu cores')
@click.option('--frequency', default=200, help='Save frequency')
def data_master(index, cores, frequency):
    """Parallel processing"""

    click.echo(f"**** Confirmed the index is: {index}")
    task_indices = gen_task_indices(index)
    task_lists = split_task(task_indices, task_nums=len(task_indices) // cores)
    # data_parser(next(iter(task_lists)), frequency)
    Parallel(n_jobs=cores)(delayed(data_parser)(tasks, frequency) for tasks in task_lists)

    send_msg(f"Well Down @ {index}!")


if __name__ == "__main__":
    data_master()
