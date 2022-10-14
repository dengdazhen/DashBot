from copy import deepcopy
import json
import numpy as np

from getFeatures import getFieldFeatures
import altair as alt
from altair_transform import extract_data
import gc


class specVectorization():
	'''
	TODO: aggregate fields features
	'''

	def __init__(self,
				 max_field_num = 10,
				 marks=['bar', 'point', 'line', 'boxplot'],
				 encodings=['x', 'y', 'color']):
		self.topic = None
		self.max_field_num = max_field_num
		self.mark_num = len(marks)
		self.marks = marks
		self.encodings = encodings
		self.cache = {}

	def load_df(self, df):
		self.df = df
		# print(self.df.columns.values)
		self.field_names, self.field_ft = getFieldFeatures(self.df)
		self.field_num, self.field_feat_len = self.field_ft.shape
		self.chart_feat_len = len(self.marks) + len(self.encodings)*self.field_feat_len
		self.mv_feat_len = self.chart_feat_len + self.max_field_num + self.max_field_num * self.field_feat_len
		self.topic_feat_index = self.chart_feat_len
		self.field_feat_index = self.chart_feat_len + self.max_field_num



	def set_topic(self, topic_name):
		self.topic = topic_name

	def __make_one_hot_mark(self, mark_type):
		'''
		return index in one-hot encoding
		'''
		assert mark_type in self.marks
		index = self.marks.index(mark_type)
		oh = np.zeros([1, self.mark_num])
		oh[0, index] = 1
		return oh

	def __make_one_hot_encoding(self, exist=True, feature=None):
		# assert exist == False or len(feature) > 0
		if exist:
			# oh[0, 0] = 1
			oh = feature[np.newaxis, :]
		else:
			oh = np.zeros([1, self.field_feat_len]) - 1
		return oh

	def __get_matched_field(self, all_names, field_name):
		for idx, name in enumerate(all_names):
			if field_name in name:
				return idx
		return -1

	def __generate_spec(self, spec):

		rp = []
		rp.append(self.__make_one_hot_mark(spec['mark']))

		newspec = deepcopy(spec)
		# try:
			# del newspec['data']
		if json.dumps(newspec) in self.cache:
			field_names, field_ft = self.cache[json.dumps(newspec)]
		elif newspec['mark'] in ['text']:
			field_names, field_ft = self.field_names, self.field_ft
		## top and bottom 5 
		elif 'transform' in newspec: 
			spec_to_cut = deepcopy(spec)
			del spec_to_cut['transform']
			try:
				res = alt.Chart.from_dict(spec_to_cut, True)
				df = extract_data(res).head(10)
				field_names, field_ft = getFieldFeatures(df)
				del df
				self.cache[json.dumps(newspec)] = [field_names, field_ft]
				if len(self.cache) % 50 == 0:
					print("cache length vector:", len(self.cache))
			except:
				print("error", newspec)
		else:
			res = alt.Chart.from_dict(spec, True)
			# print(spec)
			df = extract_data(res)
			field_names, field_ft = getFieldFeatures(df)
			del res
			del [[df]]
			gc.collect()
			self.cache[json.dumps(newspec)] = [field_names, field_ft]
			if len(self.cache) % 50 == 0:
				print("cache length vector:", len(self.cache))
		# except:
		# 	print("error here", newspec)


		for key in self.encodings:
			if key in spec['encoding'] and 'field' in spec['encoding'][key]:
				field = spec['encoding'][key]['field']
				idx = self.__get_matched_field(field_names, field)
				rp.append(self.__make_one_hot_encoding(
					exist=True, feature=field_ft[idx]))
			elif key in spec['encoding'] and 'bin' in spec['encoding'][key]:
				## for bin operation
				agg_name = 'binned'
				idx = self.__get_matched_field(field_names, agg_name)
				rp.append(self.__make_one_hot_encoding(
					exist=True, feature=field_ft[idx]))
			else:
				rp.append(self.__make_one_hot_encoding(
					exist=False, feature=None))
		if self.topic == None:
			rp.append(np.zeros([1, self.max_field_num]))
		else:
			rp_topic = np.zeros([1, self.max_field_num])
			# print(self.field_names, self.topic)
			rp_topic[0, self.field_names.index(self.topic)] = 1
			rp.append(rp_topic)
		# idx = np.argmax(np.concatenate(rp, axis=1)[0, self.topic_feat_index: self.topic_feat_index + len(self.field_names)])
		# print(np.concatenate(rp, axis=1).shape)
		for field_feature in self.field_ft:
			rp.append(field_feature[np.newaxis, :])
		# pad zeros to reach the maximum field number
		if len(self.df.columns) < self.max_field_num:
			rp.append(np.zeros([1, (self.max_field_num - len(self.df.columns))*self.field_feat_len]))
		return np.concatenate(rp, axis=1)


	def generate(self, specs):
		rps = []
		for spec in specs:
			rps.append(self.__generate_spec(spec))
		return rps

	def generate_mv_vector(self, mv_collection):
		return self.generate(mv_collection)

	def generate_param_vector(self, mv_embedding, param_type, chart_specs):
		if param_type == 'single':
			all_embeddings = []
			for chart_spec in chart_specs:
				chart_embbeding = self.__generate_spec(chart_specs)
				all_embeddings.append(np.concatenate([mv_embedding, chart_embbeding], axis = 1))
			return np.concatenate(all_embeddings, axis = 0)
		elif param_type == 'double':
			all_embeddings = []
			for chart_spec in chart_specs:
				chart_spec1, chart_spec2 =  chart_spec
				all_embeddings.append(
					np.concatenate(
						[mv_embedding, self.__generate_spec(chart_spec1), self.__generate_spec(chart_spec2)], 
						axis = 1
						)
					)
			return np.concatenate(all_embeddings, axis = 0)



if __name__ == "__main__":
	from vega_datasets import data
	from memory_profiler import profile
	import pandas as pd
	@profile
	def test():
		spec = {
			"mark": "point",
			"encoding": {
				"x": {
					"field": "Year",
					"type": "temporal"
				},
				"y": {
					"field": "Acceleration",
					"type": "quantitative"
				}
			}
		}

		source = data.cars()

		spec.update({
				'data': {'values': json.loads(source.to_json(orient='records', date_format="iso"))}})

		
		res = alt.Chart.from_dict(spec, True)
		df = extract_data(res).head(10)
		# field_names, field_ft = getFieldFeatures(df)
		res = None
		del res
		del [[df]]
		gc.collect()

	test()
	# vec = specVectorization(10)
	# vec.load_df(source)
	# print(vec.topic_feat_index)
	# print(vec.field_feat_index)
	# print(vec.field_feat_len)
	# print(vec.mv_feat_len)
	# res = vec.generate([spec])
	# print(res[0].shape)
	# print(res.shape)
