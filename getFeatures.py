import numpy as np
import pandas as pd
import math
import json
from feature_extraction.feature_extraction_func import extract_features_from_fields
from sklearn.preprocessing import normalize


def dict_data_to_array(field_data):
    list_data = []
    for item in field_data:
        list_data.append(field_data[item])
    return list_data


def getFieldFeatures(df, feature_len=82):
    fields = []
    field_names = []
    for field_name in df:
        # print(field_name)
        field_names.append(field_name)
        field_data = df[field_name].tolist()
        fields.append((
            field_name,
            {'data': field_data}
        ))
    extracted_ft = extract_features_from_fields(fields)
    # print(extracted_ft)

    features = np.zeros([len(field_names), feature_len], dtype=np.float32)
    for index, field_name in enumerate(field_names):
        for f_i, key in enumerate(extracted_ft[index]):
            # print(key)
            if extracted_ft[index][key] == True:
                features[index, f_i] = 1
            elif extracted_ft[index][key] == False:
                features[index, f_i] = 0
            elif extracted_ft[index][key] == None:
                features[index, f_i] = -1
            elif math.isnan(extracted_ft[index][key]) or abs(extracted_ft[index][key]) > 10000:
                features[index, f_i] = -1
            else:
                features[index, f_i] = extracted_ft[index][key]
    # print(features)
    features = normalize(features, axis=0)
    # print(features)
    return field_names, features


if __name__ == '__main__':
    from vega_datasets import data
    import json

    source = data.cars()
    fn, fea = getFieldFeatures(source)
    print(fn, fea)
