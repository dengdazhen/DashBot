import gym
from gym import spaces
import random

import math
import numpy as np
import pandas as pd

from specVectorization import specVectorization
from vega_datasets import data
from copy import deepcopy
import json
import altair as alt
from altair_transform import extract_data
import gc

# from draco import check_spec, dict_to_facts, dict_union, get_violations
# from draco.schema import schema_from_dataframe


class chartEnvGym(gym.Env):
	"""Custom Environment that follows gym interface"""
	metadata = {'render.modes': ['human']}

	def __init__(self, dataset, max_chart=10, max_step=100, 
            	 max_field_num = 10,
				 opts=['add', 'undo', "change", "terminate"],
				 marks=['text', 'bar',	'line',	'point', 'boxplot'],
				 encodings=['x', 'y', 'color'],
				 aggregates=[None, 'bin', 'min', 'max', 'mean',
							 'count', 'cardinality', 'top', 'bottom'],
				 ):
		'''
		action, mark, field, aggregate,
		'''
		super(chartEnvGym, self).__init__()

		assert len(dataset) > 0
		self.dataset = dataset
		self.max_field_num = max_field_num
		self.df_idx = random.choice(range(len(self.dataset)))
		self.df = self.dataset[self.df_idx]
		if len(self.df.columns) > self.max_field_num:
			self.df = self.df.iloc[:, 0:self.max_field_num]

		self.max_step = max_step
		self.vectorization = specVectorization(
			marks=marks, encodings=encodings)
		self.vectorization.load_df(self.df)
		self.fields = list(self.get_all_fields())
		self.field_map = {}
		self.init_field_map()
		self.data_json = self.df.to_dict(orient='records')
		self.state = []
		self.state_str = []
		self.opts = opts
		self.topic = None
		self.topic_id = -1
		self.marks = marks
		self.encodings = encodings
		self.aggregates = aggregates
		self.max_chart = max_chart
		self.n_step = 0
		self.max_score = 0
		self.tolerate = 0

		self.last_avg_chart_reward = 0
		self.last_mv_chart_reward = 0

		# Define action and observation space
		# They must be gym.spaces objects
		# Example when using discrete actions:
		self.action_space = spaces.MultiDiscrete(
			[len(self.opts), len(self.marks), len(self.fields), len(self.aggregates)])
		# Example for using image as input:
		self.observation_space = spaces.Box(
			low=-1, high=1, shape=(self.max_chart, self.vectorization.mv_feat_len), dtype=np.float32)
		self.history = []

		self.cache = {}

	def get_df(self):
		return self.df

	def get_rewards(self, undo=True, chart=None):
		reward = 0

		mv_reward = self.get_mv_reward() + self.get_insight_reward()
		if mv_reward > self.last_mv_chart_reward:
			reward += mv_reward - self.last_mv_chart_reward
		else:
			reward += 1.0*(mv_reward - self.last_mv_chart_reward)
		self.last_mv_chart_reward = mv_reward

		return reward

	def pad_state_vec(self, state_vec):
		if len(state_vec) > 0:
			vec = np.concatenate(state_vec, axis=0, dtype=np.float32)
			vec = np.pad(
				vec, [(self.max_chart - len(state_vec), 0), (0, 0)], mode='constant')
			return vec
		elif self.topic == None:
			vec = np.zeros(
				[self.max_chart, self.vectorization.mv_feat_len]) - 1
			return vec
		else:
			vec = np.zeros([self.max_chart, self.vectorization.mv_feat_len])
			vec[:, self.vectorization.topic_feat_index +
				self.fields.index(self.topic)+1] = 1
			return vec

	def wrap_vl_to_spec(self, vl_spec):
		new_spec = {}
		new_spec['encoding'] = []
		new_spec['type'] = vl_spec['mark']
		for key, value in vl_spec['encoding'].items():
			enc = {
				'channel': key
			}
			if 'field' in value:
				enc['field'] = value['field']
			if 'bin' in value:
				enc['bin'] = value['bin']
			if 'aggregate' in value:
				enc['aggregate'] = value['aggregate']

			new_spec['encoding'].append(enc)
		return {'mark':[new_spec]}

	def is_temporal(self, chart):
		for enc in chart['encoding']:
			if 'field' in chart['encoding'][enc] and \
				self.map_type(chart['encoding'][enc]['field']) == 'temporal':
				return True
		return False

	def is_temporal_y(self, chart):
		if 'y' in chart['encoding']:
			if 'field' in chart['encoding']['y'] and \
				self.map_type(chart['encoding']['y']['field']) == 'temporal':
				return True
		return False

	def exchange_x_y(self, old_chart):
		chart = deepcopy(old_chart)
		if 'x' in chart['encoding'] and 'y' in chart['encoding']:
			x_enc = chart['encoding']['x']
			chart['encoding']['x'] = chart['encoding']['y']
			chart['encoding']['y'] = x_enc
		return chart

	def add_bar_offset(self, old_chart):
		chart = deepcopy(old_chart)
		color_field = chart['encoding']['color']['field']
		target_field = None
		for enc in chart['encoding']:
			if enc == 'color':
				continue
			if 'bin' in chart['encoding'][enc]:
				target_field = chart['encoding'][enc]['field']
				target_enc = enc
			elif 'type' in chart['encoding'][enc] and chart['encoding'][enc]['type'] == 'nominal':
				target_field = chart['encoding'][enc]['field']
				target_enc = enc
		if target_field == None:
			return chart
		elif chart['encoding']['{}'.format(target_enc)]['field'] != color_field:
			chart['encoding']['{}Offset'.format(target_enc)] = {'field':color_field}
		return chart

	def stringify_chart(self, chart):
		new_chart = deepcopy(chart)
		if 'color' in new_chart['encoding']:
			del new_chart['encoding']['color']
		return json.dumps({
			'mark': new_chart['mark'],
			'encoding': new_chart['encoding']
		})

	def add_remark(self, chart, remark):
		if 'remark' in chart:
			chart['remark'].append(remark)
		else:
			chart['remark'] = [remark]
		return chart

	def get_insight_reward(self):

		insights = {
			'text': False,
			'temporal': False,
			'overall_distribution': False,
			'line_correlation': [],
			'point_correlation': [],
			'top_compare': [],
		}
		for chart in self.state:
			self.clean_remark(chart)


		# check temporal trend of topic
		reward = 0
		for chart in self.state:
			if insights['temporal'] == True:
				break
			if chart['mark'] not in ['line']:
				continue
			if self.is_temporal(chart):
				reward += 0.2
				remark = "Trend of {} across {} (0.2).".format(
					chart['encoding']['x']['field'], chart['encoding']['y']['field'])
				chart = self.add_remark(chart, remark)
				insights['temporal'] = True
				break


		# check topic distribution
		for chart in self.state:
			if 'y' in chart['encoding'] and 'aggregate' in chart['encoding']['y'] and \
					chart['encoding']['y']['aggregate'] == 'count':
				remark = "Data Distribution of {} (0.1).".format(chart['encoding']['x']['field'])
				chart = self.add_remark(chart, remark)
				insights['overall_distribution'] = True
				reward += 0.1
				break
		
		rev_top_bottom = {
			"top": "bottom",
			"bottom": "top"
		}
		for chart in self.state:
			if self.is_text_config(chart):
				if 'y' in chart['encoding'] and 'aggregate' in chart['encoding']['y']:
					remark = "The {} of {}.".format(chart['encoding']['y']['aggregate'],
						chart['encoding']['y']['field'])
				elif 'x' in chart['encoding'] and 'aggregate' in chart['encoding']['x']:
					remark = "The {} of {}.".format(chart['encoding']['x']['aggregate'],
						chart['encoding']['x']['field'])
				chart = self.add_remark(chart, remark)
				reward += 0.1
			elif self.is_topk_config(chart):
				# print("get top k")
				name = "{}_{}".format(chart['encoding']['y']['field'], chart['encoding']['y']['aggregate'])
				rev_name = "{}_{}".format(chart['encoding']['y']['field'], rev_top_bottom[chart['encoding']['y']['aggregate']])
				if name not in insights['top_compare'] and \
					rev_name not in insights['top_compare']:
					remark = "The {} k {} with {} {} (0.2).".format(
						chart['encoding']['y']['aggregate'],
						chart['encoding']['y']['field'],
						chart['encoding']['x']['aggregate'], chart['encoding']['x']['field'])
					chart = self.add_remark(chart, remark)
					# print("process 1", name, rev_name)
					insights['top_compare'].append(name)
					reward += 0.2
				elif name not in insights['top_compare'] and \
					rev_name in insights['top_compare']:
					# print("process 2")
					insights['top_compare'].append(name)
					remark = "In comparison with {} k ({}).".format(rev_top_bottom[chart['encoding']['y']['aggregate']], 0.3)
					chart = self.add_remark(chart, remark)
					reward += 0.3

		for chart in self.state:
			if chart['mark'] in ['point', 'line']:
				mark = chart['mark']
				try:
					f1 = chart['encoding']['x']['field']
					f2 = chart['encoding']['y']['field']

					assert self.map_type(
						f1) != "temporal" and self.map_type(f2) != "temporal"

					chart = self.clean_remark(chart)
					chart_str = self.stringify_chart(chart)
					if chart_str in self.cache:
						df = self.cache[chart_str]
					else:
						chart_a = self.append_data_altair(chart)
						# try:
						res = alt.Chart.from_dict(chart_a, True)
						df = extract_data(res)
						self.cache[chart_str] = df
						if len(self.cache) % 50 == 0:
							print("cache length env:", len(self.cache))
					# except:
					# 	continue

					pair_name = '{}_{}'.format(*sorted([f1, f2]))

					f1_cols = [[col for col in df.columns if f1 in col][0]]
					f2_cols = [[col for col in df.columns if f2 in col][0]]

					cor = df.corr()[f2_cols].loc[f1_cols].max().max()
					if pair_name not in insights['{}_correlation'.format(mark)] and np.abs(cor) > 0.6:
						insights['{}_correlation'.format(
							mark)].append((cor, pair_name))
						if len(insights['{}_correlation'.format(mark)]) == 1:
							neg = 'positive' if cor > 0 else 'negative'
							remark = '{} and {} have a high {} Pearson correlation value (0.2).'.format(
								f2, f1, neg)
							chart = self.add_remark(chart, remark)
							reward += 0.2
						elif len(insights['{}_correlation'.format(mark)]) > 1:
							neg = 'positive' if cor > 0 else 'negative'
							remark = '{} and {} also have a high {} Pearson correlation value. (0.3)'.format(
								f2, f1, neg)
							chart = self.add_remark(chart, remark)
							reward += 0.3
					del [[cor]]
					gc.collect()
					# print("step {}: cor reward:".format(self.n_step), reward)
				except:
					pass
		# print("final",reward)
		return reward

	def not_shared_field(self, chart1, chart2):
		for enc_1 in chart1['encoding']:
			if 'field' not in chart1['encoding'][enc_1]:
				continue
			field_1 = chart1['encoding'][enc_1]['field']
			for enc_2 in chart2['encoding']:
				if 'field' not in chart2['encoding'][enc_2]:
					continue
				field_2 = chart2['encoding'][enc_2]['field']
				if field_1 == field_2:
					return False
		return True

	def get_mv_reward(self):
		reward = 0
		# diversity
		# field diversity
		alpha = 5.0
		used_fields = []
		for chart in self.state:
			for enc in chart['encoding']:
				if 'field' in chart['encoding'][enc]:
					if chart['encoding'][enc]['field'] not in used_fields:
						used_fields.append(chart['encoding'][enc]['field'])
		res = 1 - math.exp(-alpha*len(used_fields)/len(self.fields))
		reward += res

		# chart type diversity
		used_types = []
		for chart in self.state:
			if chart['mark'] not in used_fields:
				used_fields.append(chart['mark'])
		res = 1 - math.exp(-alpha*len(used_types)/len(self.marks))
		reward += res

		# parsimony
		n = self.max_chart
		m = 4.0
		x = len(self.state)
		res = math.sin(math.pi*(0.5/m*min(m, x)+(max(x, m) - m)*0.5/(n-m)))
		reward += res

		return reward/3

	def valid_chart(self, chart):

		if chart['mark'] not in ['boxplot'] and not self.is_topk_config(chart):
			draco_spec = self.wrap_vl_to_spec(chart)
			if not self.check_draco_spec(draco_spec):
				return 0
		
		reward = 1
		if len(chart['encoding'].keys()) == 0:
			return 0
		## no x and no y
		if 'x' not in chart['encoding'] and 'y' not in chart['encoding']:
			return 0
		## only x or only y
		if 'x' in chart['encoding'] and 'y' not in chart['encoding']:
			if chart['encoding']['x']['type'] in ['nominal']:
				return 0
		if 'y' in chart['encoding'] and 'x' not in chart['encoding']:
			if chart['encoding']['y']['type'] in ['nominal']:
				return 0

		## aggregate or bin 'nominal' columns
		for enc in chart['encoding']:
			if 'field' not in chart['encoding'][enc]:
				continue
			field = chart['encoding'][enc]['field'] 
			if self.map_type(field) == 'nominal' and \
				('aggregate' in chart['encoding'][enc] or \
					'bin' in chart['encoding'][enc]):
				return 0
			elif self.map_type(field) == 'temporal' and \
				('aggregate' in chart['encoding'][enc]):
				return 0
		## aggregate both x and y quantitative fields
		if 'x' in chart['encoding'] and 'y' in chart['encoding']:
			if chart['encoding']['x']['type'] == 'quantitative' and \
				chart['encoding']['y']['type'] == 'quantitative':
				if 'aggregate' in chart['encoding']['x'] and 'aggregate' in chart['encoding']['y']:
					return 0
		
		## same x and y fields
		if 'x' in chart['encoding'] and 'y' in chart['encoding']:
			if 'field' in chart['encoding']['x'] and 'field' in chart['encoding']['y']:
				if chart['encoding']['x']['field'] == chart['encoding']['y']['field']:
					return 0
		
		if 'color' in chart['encoding']:
			if 'aggregate' in chart['encoding']['color'] and chart['encoding']['color']['aggregate'] in ['min', 'max', 'mean']:
				return 0
		if chart['mark'] in ['line', 'bar'] and 'x' in chart['encoding'] and 'y' in chart['encoding']:
			if chart['encoding']['x']['type'] == 'nominal' and \
				chart['encoding']['y']['type'] == 'quantitative':
				if 'aggregate' not in chart['encoding']['y']:
					return 0
			if chart['encoding']['y']['type'] == 'nominal' and \
				chart['encoding']['x']['type'] == 'quantitative':
				if 'aggregate' not in chart['encoding']['x']:
					return 0

		if chart['mark'] in ['boxplot'] and 'x' in chart['encoding'] and 'y' in chart['encoding']:
			if chart['encoding']['x']['type'] == 'quantitative' and \
				chart['encoding']['y']['type'] == 'quantitative':
				return 0

		if chart['mark'] == 'bar' and 'x' in chart['encoding'] and 'y' in chart['encoding']:
			if chart['encoding']['x']['type'] == 'quantitative' and \
				chart['encoding']['y']['type'] == 'quantitative':
				if 'aggregate' not in chart['encoding']['x'] and 'aggregate' not in chart['encoding']['y']:
					return 0
				elif 'bin' not in chart['encoding']['x'] and 'bin' not in chart['encoding']['y']:
					return 0
		
		return reward


	def gen_init_chart(self):
		return {
			'mark': None,
			'encoding': {},
		}

	def check_done(self):
		if len(self.state) == self.max_chart:
			return True
		return False

	def done(self):
		if len(self.state) == self.max_chart:
			return True
		if self.n_step >= self.max_step:
			return True
		if self.tolerate > len(self.state):
			return True
		return False

	def reset(self):
		self.df_idx = random.choice(range(len(self.dataset)))
		self.df = self.dataset[self.df_idx]
		if len(self.df.columns) > self.max_field_num:
			self.df = self.df.iloc[:, 0:self.max_field_num]

		self.state = []
		self.state_str = []
		self.history = []
		self.n_step = 0
		self.max_score = 0
		self.last_avg_chart_reward = 0
		self.last_mv_chart_reward = 0

		self.topic = None
		self.topic_id = -1
		self.vectorization.load_df(self.df)
		self.vectorization.topic = None
		self.fields = list(self.get_all_fields())		
		self.field_map = {}
		self.init_field_map()
		self.tolerate = 0

		return self.vec_state(), {"dataset_idx": self.df_idx}

	def is_text_config(self, chart):
		if chart['mark'] == 'text':
			return True
		return False

	def is_topk_config(self, chart):
		if 'y' in chart['encoding'] and \
			'aggregate' in chart['encoding']['y'] and \
			chart['encoding']['y']['aggregate'] in ['top', 'bottom']:
			return True
		return False

	def is_transform_config(self, chart):
		if 'transform' in chart:
			return True
		return False

	def get_top_or_bottom(self, chart):
		if chart['encoding']['y']['sort'] == '-x':
			return "highest"
		else: return "lowest"

	def wrap_topk_config(self, chart):
		template_top = '''{"mark": "bar","transform": [{"filter": "datum['<x_field>'] > 0"},{"window": [{"op": "rank", "as": "rank"}],"sort": [{"field": "<x_field>", "order": "descending"}]},{"window": [{"op": "row_number", "as": "row_number"}],"sort": [{"field": "<x_field>", "order": "descending"}]},{"filter":"datum.row_number < 10"}],"encoding": {"y": {"field": "<y_field>", "type": "nominal", "sort": "-x"},"x": {"field": "<x_field>", "type": "quantitative", "aggregate": "<x_agg>"}}}'''
		template_bottom= '''{"mark": "bar","transform": [{"filter": "datum['<x_field>'] > 0"},{"window": [{"op": "rank", "as": "rank"}],"sort": [{"field": "<x_field>", "order": "ascending"}]},{"window": [{"op": "row_number", "as": "row_number"}],"sort": [{"field": "<x_field>", "order": "ascending"}]},{"filter":"datum.row_number < 10"}],"encoding": {"y": {"field": "<y_field>", "type": "nominal", "sort": "x"},"x": {"field": "<x_field>", "type": "quantitative", "aggregate": "<x_agg>"}}}'''

		if chart['encoding']['y']['aggregate'] == 'top':
			x_field = chart['encoding']['x']['field']
			y_field = chart['encoding']['y']['field']
			x_agg = chart['encoding']['x']['aggregate']
			template_top = template_top.replace("<x_field>",x_field)
			template_top = template_top.replace("<y_field>",y_field)
			template_top = template_top.replace("<x_agg>",x_agg)
			return json.loads(template_top)
		elif chart['encoding']['y']['aggregate'] == 'bottom':
			x_field = chart['encoding']['x']['field']
			y_field = chart['encoding']['y']['field']
			x_agg = chart['encoding']['x']['aggregate']
			template_bottom = template_bottom.replace("<x_field>",x_field)
			template_bottom = template_bottom.replace("<y_field>",y_field)
			template_bottom = template_bottom.replace("<x_agg>",x_agg)
			return json.loads(template_bottom)
		return chart

	def make_encoding(self, enc, chart, field_id, agg_id):
		if field_id == 0:
			if self.aggregates[agg_id] in ['count']:
				chart['encoding'][enc] = {
					'type': 'quantitative',
					'aggregate': self.aggregates[agg_id]
				}
			return chart
		else:
			if self.aggregates[agg_id] == None:
				field = self.fields[field_id - 1]
				chart['encoding'][enc] = {
					'field': field,
					'type': self.map_type(field)
				}
			elif self.aggregates[agg_id] == 'bin':
				field = self.fields[field_id - 1]
				chart['encoding'][enc] = {
					'field': field,
					'type': self.map_type(field),
					'bin': True
				}
			else:
				field = self.fields[field_id - 1]
				chart['encoding'][enc] = {
					'field': field,
					'type': self.map_type(field),
					'aggregate': self.aggregates[agg_id]
				}
			return chart

	def vec_state(self):
		state_vec = self.vectorization.generate_mv_vector(
			self.return_wrapped_config_altair())
		state_vec = self.pad_state_vec(state_vec)

		return state_vec

	def step(self, actions):
		'''
		mark, (x, x_aggregate, no_x), (y, y_aggregate, no_y)
		'''
		action, x_enc, mark, y_enc, y_agg, x_agg, color_enc, color_agg = actions
		# print(self.opts[action])
		self.n_step += 1
		chart = self.gen_init_chart()
		if action == 0:  # add a new chart
			chart['mark'] = self.marks[mark]
			encodings = [
				('x', chart, self.topic_id, x_agg),
				('y', chart, y_enc, y_agg),
				('color', chart, color_enc, color_agg)
			]
			for item in encodings:  # encoding: [n_field + 1, n_agg + 1]
				chart = self.make_encoding(*item)
			if self.stringify_chart(chart) in self.state_str:
				# print("duplicated!")
				random.shuffle(self.state)
				self.state_str = [self.stringify_chart(
				chart) for chart in self.state]
				self.tolerate += 1
				reward = 0
			else:
				self.tolerate = 0
				reward = 0
				self.state.append(deepcopy(chart))
				self.state_str.append(self.stringify_chart(chart))
			reward += self.get_rewards()
			self.history.append({
				'step': self.n_step,
				'length': len(self.state),
				'reward': reward,
				'operation': 'add',
				'res': deepcopy(chart)
			})
			# print(self.state)
			return self.vec_state(), reward, self.done(), {"dataset_idx": self.df_idx}
		elif action == 1:  # undo
			# pass
			if len(self.state) > 0:
				self.state.pop()
				self.state_str.pop()
			reward = self.get_rewards()
			self.history.append({
				'step': self.n_step,
				'length': len(self.state),
				'reward': reward,
				'operation': 'undo',
				'res': None
			})
			return self.vec_state(), reward, self.done(), {"dataset_idx": self.df_idx}
		elif action == 2:  # change
			self.topic = self.fields[x_enc - 1]
			self.topic_id = x_enc
			self.vectorization.topic = self.topic
			# print("changing: ",self.df.columns.values, self.topic)
			invalid_ids = []
			for idx, chart in enumerate(self.state):
				self.state[idx]['encoding']['x']['field'] = self.topic
				self.state[idx]['encoding']['x']['type'] = self.map_type(
					self.topic)
				if 'y' in self.state[idx]['encoding'] and\
						'field' in self.state[idx]['encoding']['y'] and \
						self.topic == self.state[idx]['encoding']['y']['field']:
					invalid_ids.append(idx)
			for idx in invalid_ids[::-1]:
				self.state.pop(idx)
			self.state_str = [self.stringify_chart(
				chart) for chart in self.state]
			if self.n_step < 10:
				reward = 0
			else:
				reward = self.get_rewards()
			self.history.append({
				'step': self.n_step,
				'length': len(self.state),
				'reward': reward,
				'operation': 'change',
				'res': self.topic
			})
			# print(self.state)
			return self.vec_state(), reward, self.done(), {"dataset_idx": self.df_idx}
		elif action == 3:  # early terminate
			self.history.append({
				'step': self.n_step,
				'length': len(self.state),
				'reward': 0.3,
				'operation': 'terminate',
				'res': None
			})
			if self.done():
				return self.vec_state(),0, True, {"dataset_idx": self.df_idx}
			else:
				if len(self.state) < 4:
					random.shuffle(self.state)
					self.tolerate += 1
					self.state_str = [self.stringify_chart(
				chart) for chart in self.state]
					return self.vec_state(), 0, self.done(), {"dataset_idx": self.df_idx}
				else:
					return self.vec_state(), 0.3, True, {"dataset_idx": self.df_idx}

	def get_state(self):
		return self.state

	def append_data(self, chart):
		new_chart = deepcopy(chart)
		if self.is_topk_config(new_chart):
			new_chart = self.wrap_topk_config(new_chart)
		new_chart.update({
			'data': {'name': 'values'}})
		return new_chart

	def append_data_altair(self, chart):
		new_chart = deepcopy(chart)
		if self.is_topk_config(new_chart):
			new_chart = self.wrap_topk_config(new_chart)
		self.clean_remark(new_chart)
		new_chart.update({
			'data': {'values': json.loads(self.df.to_json(orient='records', date_format="iso"))}})
		return new_chart

	def add_title(self, chart):
		if 'y' in chart['encoding']:
			x_field = chart['encoding']['x']['field']
			if not self.is_transform_config(chart) and 'field' in chart['encoding']['y']:
				y_field =  chart['encoding']['y']['field']
				title = "{} vs. {}".format(y_field, x_field)
			elif self.is_transform_config(chart):
				y_field =  chart['encoding']['y']['field']
				top_or_bottom = self.get_top_or_bottom(chart)
				title = "{} with {} {}".format(y_field, top_or_bottom, x_field)
			else:
				title = "distribution of {}".format(x_field)
			chart['title'] = title
		else:
			pass

	def return_wrapped_config(self):
		new_charts = []
		for chart in self.state:
			if chart['mark'] == 'line':
				chart = self.exchange_x_y(chart)
			elif chart['mark'] == 'bar' and 'color' in chart['encoding']:
				chart = self.add_bar_offset(chart)
			new_chart = self.append_data(chart)
			self.add_title(new_chart)
			new_charts.append(new_chart)
		return new_charts

	def return_wrapped_config_single(self, chart):
		if chart['mark'] == 'line':
			chart = self.exchange_x_y(chart)
		elif chart['mark'] == 'bar' and 'color' in chart['encoding']:
			chart = self.add_bar_offset(chart)
		new_chart = self.append_data(chart)
		self.add_title(new_chart)
		return new_chart

	def return_wrapped_config_altair(self):
		return [self.append_data_altair(chart) for chart in self.state]


	def update_fields(self, new_fields):
		for field in new_fields:
			if field['type'] != self.map_type(field['value']):
				print("update field type")
				self.field_map[field['value']] = field['type']

	def get_data(self):
		data_list = json.loads(self.df.to_json(orient='records'))
		return {
			'values': data_list
		}

	def get_used_encodings(self):
		encodings = []
		for encoding in self.current_state['encoding']:
			encodings.append(encoding)
		return encodings

	def get_all_fields(self):
		return self.df.columns.values

	def translate_type(self, field):
		### TODO: ordinal
		dtype = self.df[field].dtype
		if dtype in ['datetime64[ns]'] or \
				field.lower() in ['year', 'years', 'date', 'time']:
			return 'temporal'
		elif dtype in ['object', 'str', 'category', 'bool']:
			return 'nominal'
		else:
			return 'quantitative'

	def map_type(self, field):
		return self.field_map[field]

	def init_field_map(self):
		for field in self.fields:
			self.field_map[field] = self.translate_type(field)

	def is_top_or_bottom_transform(self, chart):
		if chart['transform'][1]['sort'][0]['order'] == 'descending':
			return 'top'
		else:
			return 'bottom'

	def clean_remark(self, chart):
		keys = ['remark', 'title', 'pos']
		for key in keys:
			if key in chart:
				del chart[key]
		return chart

	def transform_back(self, chart):
		chart = self.clean_remark(chart)
		if 'transform' in chart:
			if self.is_top_or_bottom_transform(chart) == 'top':
				nchart = {
					'mark': 'bar',
					'encoding':{
						'x': {
							'field': chart["encoding"]["x"]["field"], 
							'type': chart["encoding"]["x"]["type"],
							'aggregate': chart["encoding"]["x"]["aggregate"]},
						'y': {'field':chart["encoding"]["y"]["field"] , 
						'type': chart["encoding"]["y"]["type"], 
							'aggregate':"top"}
					}
				}
			else:
				nchart = {
					'mark': 'bar',
					'encoding':{
						'x': {
							'field': chart["encoding"]["x"]["field"], 
							'type': chart["encoding"]["x"]["type"],
							'aggregate': chart["encoding"]["x"]["aggregate"]},
						'y': {'field':chart["encoding"]["y"]["field"] , 
						'type': chart["encoding"]["y"]["type"], 
							'aggregate':"bottom"}
					}
				}
		elif chart['mark'] == 'line':
			nchart = self.exchange_x_y(chart)
		else:
			nchart = chart
		return nchart

	def update_state(self, new_state):
		self.reset()
		for chart in new_state['states']:
			nchart = self.transform_back(chart)
			if self.stringify_chart(nchart) in self.state_str:
				continue
			else:
				self.state.append(nchart)
				self.state_str.append(self.stringify_chart(nchart))
		
		return True

	def recommend(self, candidates):
		all_res = []
		all_res_str = []
		reward = self.get_rewards()
		for chart in candidates:
			nchart = self.transform_back(chart)
			if self.stringify_chart(nchart) not in self.state_str and \
				self.stringify_chart(nchart) not in all_res_str:
				self.state.append(nchart)
				new_reward = self.get_rewards()
				all_res_str.append(self.stringify_chart(nchart))
				all_res.append({
					"return_dif":new_reward - reward,
					"configure": self.return_wrapped_config_single(nchart)
				})
			else:
				continue
		return list(sorted(all_res, key=lambda x: x['return_dif'], reverse=True))

	def return_fields(self):
		field_info = []
		for field in self.fields:
			field_info.append({
				'value': field,
				'label': field,
				'type': self.map_type(field)
			})
		return field_info


if __name__ == '__main__':
	data = data.cars()
	env = chartEnvGym(data)
	print(list(env.fields).index('Miles_per_Gallon'))
