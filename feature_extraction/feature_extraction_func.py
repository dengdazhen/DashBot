#!/usr/bin/env python
# coding: utf-8


import traceback
import pandas as pd

import os
import json
from time import time, strftime

from collections import OrderedDict

import sys
sys.path.append("./feature_extraction")

from feature_extraction.features.single_field_features import extract_single_field_features
from feature_extraction.outcomes.field_encoding_outcomes import extract_field_outcomes
from feature_extraction.general_helpers import load_raw_data, clean_chunk

import pickle
import numpy as np
from sklearn.model_selection import train_test_split


MAX_FIELDS = 25
total_charts = 0
charts_without_data = 0
chart_loading_errors = 0
feature_extraction_errors = 0
charts_exceeding_max_fields = 0
CHUNK_SIZE = 1000


def extract_features_from_fields(fields, chart_obj={}, num_fields=2):
    results = {}


    single_field_features, parsed_fields = extract_single_field_features(
            fields, MAX_FIELDS=MAX_FIELDS, num_fields=num_fields)

    df_field_level_features = []
    for i, f in enumerate(single_field_features):
        if f['exists']:
            df_field_level_features.append(f)
        
    
    # results['df_field_level_features'] = df_field_level_features

    # field_level_outcomes = extract_field_outcomes(chart_obj)
    # results['field_outcomes'] = list(
    #     list(field_level_outcomes)[0].keys())
    # results['df_field_level_outcomes'] = field_level_outcomes

    return df_field_level_features

