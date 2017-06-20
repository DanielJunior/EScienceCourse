import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem.snowball import SnowballStemmer
import nltk
import re
import xml.etree.ElementTree as ET
from graphviz import Digraph
from prov_recovery_database import ProvRecoveryDatabase
import datetime


class ProvRecoveryModel():
    STOPWORDS = "english"

    def __init__(self, dataset_path, max_features=1000, stopwords="english", clusters_numbers=100):
        self.dataset_path = dataset_path
        self.max_features = max_features
        ProvRecoveryModel.STOPWORDS = stopwords
        self.clusters_numbers = clusters_numbers
        nltk.download('punkt')
        nltk.download('snowball_data')
        nltk.download('stopwords')

    def train(self):
        self.parse_dataset()
        self.tranformed_data = self.transform_dataset()
        self.kmeans_model = self.generate_model(self.tranformed_data)

    def predict(self, xml_path):
        parsed_data = self.extract_abstract_from_xml(xml_path)
        tranformed_data = self.transform_data(parsed_data)
        predict = self.kmeans_model.predict(tranformed_data)
        return predict

    def generate_lineage(self, xml_path):
        prediction = self.predict(xml_path)
        candidates_idx = [i for i, x in enumerate(self.kmeans_model.labels_) if x == prediction]
        candidates_filenames = [self.filenames[x] for x in candidates_idx]
        # print("Print unordered candidates: %a" % candidates_filenames)
        candidates_filenames.sort(key=lambda x: os.stat(x).st_mtime)
        # print("Print ordered candidates: %a" % candidates_filenames)
        self.generate_provenance_chart(xml_path, candidates_filenames)
        self.generate_provenance_database(candidates_filenames)

    def generate_provenance_chart(self, name, ordered_filenames):
        dot = Digraph(comment=str(name) + ' provenance chart')
        for filename in ordered_filenames:
            dot.node(filename, filename)
        for idx, filename in enumerate(ordered_filenames):
            if idx < len(ordered_filenames) - 1:
                dot.edge(ordered_filenames[idx], ordered_filenames[idx + 1], constraint='false')
        dot.render('data/output/' + str(name.split("/")[-1]) + "_provenance_chart", view=True)

    def generate_provenance_database(self, ordered_filenames):
        database = ProvRecoveryDatabase()
        database.open_database()
        for filename in ordered_filenames:
            database.insert_into_entity(filename, datetime.datetime.fromtimestamp(os.stat(filename).st_mtime))
        for idx, filename in enumerate(ordered_filenames):
            if idx > 0:
                database.insert_into_was_derived(filename, ordered_filenames[idx - 1])
            else:
                database.insert_into_was_derived(filename, None)

    def transform_data(self, parsed_data):
        return self.tfidf_vectorizer.transform([parsed_data])

    def transform_dataset(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words=self.STOPWORDS, max_features=self.max_features,
                                                tokenizer=ProvRecoveryModel.tokenize_and_stem, )
        return self.tfidf_vectorizer.fit_transform(self.parsed_data).todense()

    def tokenize_and_stem(text):
        stemmer = SnowballStemmer(ProvRecoveryModel.STOPWORDS)
        tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        stems = [stemmer.stem(t) for t in filtered_tokens]
        return stems

    def generate_model(self, data):
        km = KMeans(n_clusters=self.clusters_numbers, init='k-means++', max_iter=100, n_init=1, verbose=False)
        km.fit(data)
        return km

    def parse_dataset(self):
        self.filenames = []
        self.parsed_data = []
        for data_path, dirs, files in os.walk(self.dataset_path):
            for filename in files:
                fullpath = os.path.join(data_path, filename)
                self.filenames.append(fullpath)
                data = self.extract_abstract_from_xml(fullpath)
                if not data == None:
                    self.parsed_data.append(data)

    def extract_abstract_from_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        abstract = root.find('abstract')
        if not abstract == None:
            return abstract.text
        else:
            return None
