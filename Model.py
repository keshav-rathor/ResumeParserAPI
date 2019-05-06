import json
import logging
from tqdm import tqdm
import spacy
import plac #  wrapper over argparse
import random
from pathlib import Path
import numpy as np
from itertools import compress
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from sklearn.metrics import accuracy_score
import re

# Function to convert data format from dataurks to as per Spacy needs
def convert_dataturks_to_spacy(dataturks_JSON_FilePath):
    try:
        training_data = []
        lines=[]
        with open(dataturks_JSON_FilePath, 'r',  encoding="utf8") as f:
            lines = f.readlines()

        for i,line in enumerate(lines):
            # print(i)
            data = json.loads(line)
            text = data['content']
            entities = []
            if data['annotation']:
                for annotation in data['annotation']:
                    #only a single point in text annotation.
                    point = annotation['points'][0]
                    labels = annotation['label']
                    # handle both list of labels or a single label.
                    if not isinstance(labels, list):
                        labels = [labels]

                    for label in labels:
                        #dataturks indices are both inclusive [start, end] but spacy is not [start, end)
                        entities.append((point['start'], point['end'] + 1 ,label))

            training_data.append((text, {"entities" : entities}))

        return training_data
    except Exception as e:
        logging.exception("Unable to process " + dataturks_JSON_FilePath + "\n" + "error = " + str(e))
        return None

# Function to train Spacy NER
def train_spacy(data, output_dir=None, model = None, n_iter=100):
    TRAIN_DATA = convert_dataturks_to_spacy(data)
    num_entities = np.array([len(TRAIN_DATA[i][1]['entities']) for i in range(len(TRAIN_DATA))])
    TRAIN_DATA = list(compress(TRAIN_DATA, num_entities != 0))

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy

    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

        # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        try:
            optimizer = nlp.begin_training()
            for itn in range(n_iter):
                print("Statring iteration " + str(itn))
                random.shuffle(TRAIN_DATA)
                losses = {}
                for text, annotations in tqdm(TRAIN_DATA):
                    nlp.update(
                        [text],  # batch of texts
                        [annotations],  # batch of annotations
                        drop=0.2,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses)
                print(losses)
        except MemoryError:
            print("MemoryError")
            pass

        # save model to output directory
        if output_dir is not None:
            output_path = Path(output_dir)
            if not output_path.exists():
                output_path.mkdir()
            nlp.to_disk(output_path)
            print("Saved model to", output_path)

# Training NER model
train_spacy(data="./Resume Skill Annotation v2.json",
            output_dir="./resume_parser_skills",
            n_iter=100)

