from utils.read_write_data import read_json, makedir, save_dict, write_txt
import argparse
from collections import namedtuple
import os
import nltk
from nltk.tag import StanfordPOSTagger
from random import shuffle
import numpy as np
import pickle
import transformers as ppb
import time
import json

reid_raw_data = read_json('./nouns_10_choose.json')

print(len(reid_raw_data.keys()))