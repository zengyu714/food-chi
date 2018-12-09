"""
Ref: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/word2vec.ipynb
"""
import json
import time
from operator import itemgetter
from pathlib import Path

import gensim
import logging
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
from similarity.jarowinkler import JaroWinkler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from preprocess import map_to_dictionary
from utils.tools import is_chinese
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


class Word2VecModel:
    def __init__(self):
        self.train_user_recipes, self.test_user_recipes = self._load_users()
        self.min_count = 3
        self.model = None

    @staticmethod
    def _load_users():
        recipes = []
        for json_file in Path("data/preprocessed/").glob("mapped_starred*"):
            for val in json.load(json_file.open(encoding="utf-8")).items():
                stripped = val[0], [v.strip() for v in val[1]]
                recipes.append(stripped)
        return train_test_split(recipes, test_size=0.20, random_state=42)

    @staticmethod
    def _gen_values(list_object):
        return list(map(itemgetter(1), list_object))

    @staticmethod
    def _cal_precision_recall(query, true, pred, topn=10):
        true = set(true) - set(query)
        pred = set(pred[:topn])
        intersect = len(true.intersection(pred))
        precision = intersect / (len(true) + 1e-4)
        recall = intersect / (len(pred) + 1e-4)
        return precision, recall

    def _load_dictionary(self):
        self.zh2en = json.load(open("data/preprocessed/dictionary.json", encoding="utf-8"))
        self.en2zh = {v: k for k, v in self.zh2en.items()}

    def _load_models(self):
        if self.model is None:
            model_path = sorted(Path("models").glob("word2vec*.model"), reverse=True)[0]
            self.model = gensim.models.Word2Vec.load(str(model_path))  # open the model

    def train(self):
        """Train models on these user recipes."""

        self.model = gensim.models.Word2Vec(self._gen_values(self.train_user_recipes), compute_loss=True, sg=1,
                                            min_count=self.min_count, seed=14)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        print(self.model.get_latest_training_loss())

        self.model.save(f"models/word2vec_{stamp}.model")

    def plot_tsne(self, num_dimensions=2):
        vectors = []  # positions in vector space
        labels = []  # keep track of words to label our data again later
        self._load_models()
        self._load_dictionary()

        for word in self.model.wv.vocab:
            try:
                vectors.append(self.model[word])
                labels.append(f"{self.zh2en[word]}({word})")
            except KeyError:
                continue

        # convert both lists into numpy vectors for reduction
        vectors = np.asarray(vectors)
        labels = np.asarray(labels)

        # reduce using t-SNE
        vectors = np.asarray(vectors)
        logging.info('Starting tSNE dimensionality reduction. This may take some time.')
        tsne = TSNE(n_components=num_dimensions, random_state=0)
        vectors = tsne.fit_transform(vectors)

        x_vals = [v[0] for v in vectors]
        y_vals = [v[1] for v in vectors]

        # Create a trace
        trace = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='text',
            text=labels
        )

        data = [trace]
        logging.info('All done. Plotting.')
        plot(data, filename='res/word_tsne.html')

    def evaluate(self):
        self._load_models()

        hits_at_5, hits_at_10 = [], []
        for user, recipes in self.test_user_recipes:
            if len(recipes) < self.min_count:
                continue
            for query in recipes:
                if query not in self.model.wv.vocab:
                    continue
                recommend = self.model.wv.similar_by_word(query, topn=10)
                hits_at_5.append(self._cal_precision_recall(query, recipes, recommend, topn=5))
                hits_at_10.append(self._cal_precision_recall(query, recipes, recommend, topn=10))

        precision_5, recall_5 = np.array(hits_at_5).mean(axis=0)
        precision_10, recall_10 = np.array(hits_at_10).mean(axis=0)

        print(f"Precision @5: {precision_5}\t Recall @5: {recall_5}")
        print(f"Precision @10: {precision_10}\t Recall @10: {recall_10}")

    def predict(self, queries="Noodles Mushroom Noodles"):
        self._load_models()
        self._load_dictionary()

        if not isinstance(queries, (list, tuple)):
            queries = [queries]

        jarowinkler = JaroWinkler()

        for query in queries:
            print(f"Input: {query}")
            # First map input to dictionary
            mapping = self.zh2en.keys() if is_chinese(query) else self.zh2en.values()
            query = map_to_dictionary(jarowinkler, query, mapping)
            # Need convert to Chinese if the input is English
            if not is_chinese(query):
                query = self.en2zh[query]
            results = self.model.wv.similar_by_word(query, topn=10)
            for idx, (name_zh, score) in enumerate(results, start=1):
                print(f"#{idx} - {self.zh2en[name_zh]}: {score:.3f}")


if __name__ == "__main__":
    model = Word2VecModel()
    # model.train()
    # model.plot_tsne()
    # model.predict("Sweet and sour fish")
    while True:
        keywords = input(f"\n==> Please enter your heuristic keywords (e.g. sweet and sour fish) [q to quit]: ")
        if keywords == 'q':
            break
        else:
            model.predict(keywords)
    print("\nHave a nice day!")