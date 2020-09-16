import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_index_from_name(name):
	return data_frame[data_frame.name == name]['index'].values[0]


def get_name_from_index(index):
	return data_frame[data_frame.index == index]['name'].values[0]


# import data
data_frame = pd.read_csv("profile_data_set.csv")
# data_frame.drop("age",axis = 1)

features = ['job','interest']

def feature_mixer(row):
	return row['job'] + " " + row['interest']

data_frame["mixed_features"] = data_frame.apply(feature_mixer,axis=1)


cv = CountVectorizer()
metrix = cv.fit_transform(data_frame["mixed_features"])


cosine_sim = cosine_similarity(metrix)
requred_name = 'abhijith'

name_index = get_index_from_name(requred_name)
similer_name = list(enumerate(cosine_sim[name_index]))

sorted_list = sorted(similer_name,key=lambda x:x[1],reverse = True)

i = 0
for person in sorted_list:
	print(get_name_from_index(person[0]))
	i = i + 1
	if i > 3:
		break
