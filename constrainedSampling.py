from copy import deepcopy
from jinja2 import ChoiceLoader
from matplotlib.style import available
from threading import enumerate
from typing import Optional
import torch
import pfrl
from pfrl.policies import SoftmaxCategoricalHead
import torch
from torch.distributions.categorical import Categorical
from torch import einsum
from einops import  reduce
import numpy as np


class CategoricalMasked(Categorical):

    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self.mask = mask
        self.batch, self.nb_action = logits.size()
        if mask is None:
            super(CategoricalMasked, self).__init__(logits=logits)
        else:
            self.mask_value = torch.finfo(logits.dtype).min
            logits.masked_fill_(~self.mask, self.mask_value)
            super(CategoricalMasked, self).__init__(logits=logits)

    def entropy(self):
        if self.mask is None:
            return super().entropy()
        # Elementwise multiplication
        p_log_p = einsum("ij,ij->ij", self.logits, self.probs)
        # Compute the entropy with possible action only
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device),
        )
        return -reduce(p_log_p, "b a -> b", "sum", b=self.batch, a=self.nb_action)

class consSampling():
	def __init__(self, dataset, max_field_num, topic_feat_index, field_feat_index, field_feat_len,
			opts = ['add', 'undo', "change", "terminate"],
			marks = ['text', 'bar',	'line',	'point', 'boxplot'],
			encodings = ['x', 'y', 'color'],
			aggregates = [None, 'bin','min','max','mean','count', 'cardinality', 'top', 'bottom']):
		self.field_feat_index = field_feat_index
		self.field_feat_len = field_feat_len
		self.max_field_num = max_field_num
		self.dataset = dataset
		# print(self.fields)
		self.topic_feat_index = topic_feat_index
		self.opts = opts
		self.marks = marks
		self.encodings = encodings
		self.aggregates = aggregates
		self.quantitative_agg_idx = [2,3,4]
		self.count_agg_idx = [5]
		self.null_agg_idx = [0]
		self.nominal_agg_idx = [1]

	def set_df(self, idx):
		self.df = self.dataset[idx]
		self.fields = list(self.df.columns.values)


	def is_no_topic(self, state):
		if state[-1,self.topic_feat_index] == -1:
			return True
		return False


	def get_topic_index(self, state):
		field_one_hot = state[-1,self.topic_feat_index: self.topic_feat_index + len(self.fields)]
		# print(field_one_hot)
		return np.argmax(field_one_hot)

	def is_empty_mv(self, state):
		## all chart vectors are 0 padded
		if np.sum(state) == state.shape[0]:
			return True
		return False

	def is_less_mv(self, state):
		## all chart vectors are 0 padded
		if np.sum(np.sum(state, axis = 1) == 0) < 3:
			return True
		return False

	def make_mask(self, one_index, total_length):
		mask = torch.zeros([1, total_length], dtype = torch.bool)
		mask[0, one_index] = True
		return mask

	def make_full_mask(self, total_length):
		mask = torch.ones([1, total_length], dtype = torch.bool)
		return mask

	# def empty

	def cut_branch(self, digit, itemlist, padded_size = None):
		if padded_size is not None:
			mask = self.make_mask([], padded_size)
		else:
			mask = self.make_mask([], len(itemlist))
		# print(digit, mask)
		pout = CategoricalMasked(digit.unsqueeze(0), mask)
		val = torch.tensor([0])
		return pout, val

	def extend_by_choice(self, digit, items, itemlist, padded_size = None):
		# print(items, itemlist)
		if len(items) == 0:
			return self.cut_branch(digit, itemlist, padded_size = padded_size)
		if len(items) == len(itemlist):
			if padded_size is not None:
				mask = self.make_full_mask(padded_size)
			else:
				mask = self.make_full_mask(len(itemlist))
		else:
			index_list = []
			for c in items:
				index_list.append(itemlist.index(c))
			# print(index_list)			
			if padded_size is not None:
				mask = self.make_mask(index_list, padded_size)
			else:
				mask = self.make_mask(index_list, len(itemlist))
		# print(digit.shape, mask.shape)
		pout = CategoricalMasked(digit.unsqueeze(0), mask)
		val = pout.sample()
		return pout, val

	def chained_sampling_batch(self, statevar, avecs):
		# print(avecs[0].shape)
		para_num = 8
		pout_logits = [[] for i in range(para_num)]
		action_samples = [[] for i in range(para_num)]
		for batch_i in range(statevar.shape[0]):
			state = statevar.numpy()[batch_i]
			avec = [avecs[i][batch_i] for i in range(para_num)]
			pout, action = self.chained_sampling(state, avec)
			for i in range(para_num):
				pout_logits[i].append(pout[i].mask)
				action_samples[i].append(action[i])
		pouts = []
		actions = []
		for i in range(para_num):
			# print(torch.concat(pout_logits[i]).shape)
			pouts.append(CategoricalMasked(avecs[i],torch.concat(pout_logits[i])))
			actions.append(torch.concat(action_samples[i]))
		return pouts, actions

	def get_cardinality(self, field):
		# print(field, self.fields.index(field), len(self.df[field].unique()))
		return len(self.df[field].unique())

	def get_defields(self, field_name):
		new_fields = deepcopy(self.fields)
		new_fields.remove(field_name)
		return new_fields
			
			

	def chained_sampling(self, state, avec):
		act, x_topic_enc, mark, y_enc, y_agg, x_topic_agg, color_enc, color_agg = avec

		if self.is_no_topic(state):
			act_pout, act_val = self.extend_by_choice(act, ["change"], self.opts)
		elif self.is_empty_mv(state) or self.is_less_mv(state):
			act_pout, act_val = self.extend_by_choice(act, ["add"], self.opts)
		else:
			act_pout, act_val = self.extend_by_choice(act,  self.opts, self.opts)

		if self.opts[act_val] == 'terminate':
			x_topic_enc_pout, x_topic_enc_val = self.cut_branch(x_topic_enc, [None] + self.fields, 1 + self.max_field_num)
			mark_pout, mark_val = self.cut_branch(mark, self.marks)
			y_enc_pout, y_enc_val = self.cut_branch(y_enc, [None] + self.fields, 1 + self.max_field_num)
			y_agg_pout, y_agg_val = self.cut_branch(y_agg, self.aggregates)
			x_topic_agg_pout, x_topic_agg_val = self.cut_branch(x_topic_agg, self.aggregates)
			color_enc_pout, color_enc_val = self.cut_branch(color_enc, [None] + self.fields, 1 + self.max_field_num)
			color_agg_pout, color_agg_val = self.cut_branch(color_agg, self.aggregates)
		elif self.opts[act_val] == 'change':
			if self.is_no_topic(state):
				x_topic_enc_pout, x_topic_enc_val = self.extend_by_choice(x_topic_enc, 
						[f for f in self.fields if self.translate_type(f) in ['quantitative']], [None] + self.fields, 1 + self.max_field_num)
			else:
				field_id = self.get_topic_index(state)
				field_name = self.fields[field_id]
				# print(state, state.shape, field_name, self.get_topic_index(state))
				x_topic_enc_pout, x_topic_enc_val = self.extend_by_choice(x_topic_enc, 
						[f for f in self.fields if self.translate_type(f) in ['quantitative'] and f != field_name], [None] + self.fields, 1 + self.max_field_num)
			mark_pout, mark_val = self.cut_branch(mark, self.marks)
			y_enc_pout, y_enc_val = self.cut_branch(y_enc, [None] + self.fields, 1 + self.max_field_num)
			y_agg_pout, y_agg_val = self.cut_branch(y_agg, self.aggregates)
			x_topic_agg_pout, x_topic_agg_val = self.cut_branch(x_topic_agg, self.aggregates)
			color_enc_pout, color_enc_val = self.cut_branch(color_enc, [None] + self.fields, 1 + self.max_field_num)
			color_agg_pout, color_agg_val = self.cut_branch(color_agg, self.aggregates)
		elif self.opts[act_val] in ['add', 'undo']:
			x_topic_enc_pout, x_topic_enc_val = self.cut_branch(x_topic_enc, [None] + self.fields, 1 + self.max_field_num)
			field_id = self.get_topic_index(state)
			x_field = self.fields[field_id]

			if self.opts[act_val] in ['undo']:
				mark_pout, mark_val = self.cut_branch(mark, self.marks)
				y_enc_pout, y_enc_val = self.cut_branch(y_enc, [None] + self.fields, 1 + self.max_field_num)
				y_agg_pout, y_agg_val = self.cut_branch(y_agg, self.aggregates)
				x_topic_agg_pout, x_topic_agg_val = self.cut_branch(x_topic_agg, self.aggregates)
				color_enc_pout, color_enc_val = self.cut_branch(color_enc, [None] + self.fields, 1 + self.max_field_num)
				color_agg_pout, color_agg_val = self.cut_branch(color_agg, self.aggregates)
			elif self.opts[act_val] in ['add']:
				mark_pout, mark_val = self.extend_by_choice(mark, self.marks, self.marks)
				if self.marks[mark_val] == 'text':
					y_enc_pout, y_enc_val = self.extend_by_choice(y_enc, [None] + self.get_defields(x_field), [None] + self.fields, 1 + self.max_field_num)
					if y_enc_val == 0:
						y_agg_pout, y_agg_val = self.cut_branch(y_agg, self.aggregates)
						x_topic_agg_pout, x_topic_agg_val = self.extend_by_choice(x_topic_agg, ['min', 'max', 'mean'], self.aggregates)
					# x quantitative, y nominal
					elif self.translate_type(self.fields[y_enc_val - 1]) in ['nominal', 'temporal']:
						y_agg_pout, y_agg_val = self.extend_by_choice(y_agg, ['cardinality'], self.aggregates)
						x_topic_agg_pout, x_topic_agg_val = self.cut_branch(x_topic_agg, self.aggregates)
					# x quantitative, y quantitative
					elif self.translate_type(self.fields[y_enc_val - 1]) in ['quantitative']:
						# print("bar", self.translate_type(self.fields[y_enc_val - 1]), self.fields[y_enc_val - 1], x_field, self.fields, field_id, np.sum(state), self.is_no_topic(state), act_val)
						y_agg_pout, y_agg_val = self.extend_by_choice(y_agg, ['min', 'max', 'mean'], self.aggregates)
						x_topic_agg_pout, x_topic_agg_val = self.cut_branch(x_topic_agg, self.aggregates)
				# bar chart
				elif self.marks[mark_val] == 'bar':
					y_enc_pout, y_enc_val = self.extend_by_choice(y_enc, [None] + self.get_defields(x_field), [None] + self.fields, 1 + self.max_field_num)
					if y_enc_val == 0:
						y_agg_pout, y_agg_val = self.extend_by_choice(y_agg, ['count'], self.aggregates)
						x_topic_agg_pout, x_topic_agg_val = self.extend_by_choice(x_topic_agg, ['bin'], self.aggregates)
					# x quantitative, y nominal
					elif self.translate_type(self.fields[y_enc_val - 1]) in ['nominal', 'temporal']:
						# print("comes here")
						if self.get_cardinality(self.fields[y_enc_val - 1]) <= 20:
							# print("comes here 1")
							y_agg_pout, y_agg_val = self.extend_by_choice(y_agg, [None], self.aggregates)
						else:
							# print("comes here 2")
							y_agg_pout, y_agg_val = self.extend_by_choice(y_agg, ['top','bottom'], self.aggregates)
						x_topic_agg_pout, x_topic_agg_val = self.extend_by_choice(x_topic_agg, ['mean'], self.aggregates)
					# x quantitative, y quantitative
					elif self.translate_type(self.fields[y_enc_val - 1]) in ['quantitative']:
						# print("bar", self.translate_type(self.fields[y_enc_val - 1]), self.fields[y_enc_val - 1], x_field, self.fields, field_id, np.sum(state), self.is_no_topic(state), act_val)
						y_agg_pout, y_agg_val = self.extend_by_choice(y_agg, ['bin','mean'], self.aggregates)
						# x quantitative, y agg quantitative
						if self.aggregates[y_agg_val] in ['mean']:
							x_topic_agg_pout, x_topic_agg_val = self.extend_by_choice(x_topic_agg, ['bin'], self.aggregates)
						elif self.aggregates[y_agg_val] in ['bin']:
							x_topic_agg_pout, x_topic_agg_val = self.extend_by_choice(x_topic_agg, ['mean'], self.aggregates)
				# line chart
				elif self.marks[mark_val] == 'line':
					# remove bin count for line chart
					available_fields = []
					for f in self.fields:
						if self.translate_type(f) in ['temporal','quantitative'] and f != x_field:
							available_fields.append(f)
					y_enc_pout, y_enc_val = self.extend_by_choice(y_enc, available_fields, [None] + self.fields, 1 + self.max_field_num)
					# # x quantitative, no y field
					# if y_enc_val == 0:
					# 	y_agg_pout, y_agg_val = self.extend_by_choice(y_agg, ['count'], self.aggregates)
					# 	x_topic_agg_pout, x_topic_agg_val = self.extend_by_choice(x_topic_agg, ['bin'], self.aggregates)
					# x quantitative, y nominal
					# if self.translate_type(self.fields[y_enc_val - 1]) in ['temporal']:
					y_agg_pout, y_agg_val = self.extend_by_choice(y_agg, [None], self.aggregates)
					x_topic_agg_pout, x_topic_agg_val = self.extend_by_choice(x_topic_agg, ['mean'], self.aggregates)
					# x quantitative, y quantitative
					# elif self.translate_type(self.fields[y_enc_val - 1]) in ['quantitative']:
						# y_agg_pout, y_agg_val = self.extend_by_choice(y_agg, ['min','max','mean'], self.aggregates)
						# x quantitative, y agg quantitative
						# if self.aggregates[y_agg_val] in ['min','max','mean']:
						# x_topic_agg_pout, x_topic_agg_val = self.extend_by_choice(x_topic_agg, ['bin'], self.aggregates)
						# elif self.aggregates[y_agg_val] in ['bin']:
						# 	x_topic_agg_pout, x_topic_agg_val = self.extend_by_choice(x_topic_agg, ['min', 'max', 'mean'], self.aggregates)
				# scatterplot
				elif self.marks[mark_val] == 'point':
					available_fields = []
					for f in self.fields:
						if self.translate_type(f) in ['quantitative'] and f != x_field:
							available_fields.append(f)
					y_enc_pout, y_enc_val = self.extend_by_choice(y_enc, available_fields, [None] + self.fields, 1 + self.max_field_num)
					# print(self.is_no_topic(state), self.is_empty_mv(state), np.sum(state), state.shape[0], x_field, self.fields[y_enc_val - 1])
					# x quantitative, y quantitative
					if self.translate_type(self.fields[y_enc_val - 1]) in ['quantitative']:
						y_agg_pout, y_agg_val = self.cut_branch(y_agg, self.aggregates)
						x_topic_agg_pout, x_topic_agg_val = self.cut_branch(x_topic_agg, self.aggregates)
				# boxplot
				elif  self.marks[mark_val] == 'boxplot':
					available_fields = []
					for f in self.fields:
						if f != x_field and self.translate_type(f) in ['nominal'] and self.get_cardinality(f) <= 5:
							available_fields.append(f)
						elif  f != x_field and self.translate_type(f) in ['temporal'] and self.get_cardinality(f) <= 12:
							available_fields.append(f)
					y_enc_pout, y_enc_val = self.extend_by_choice(y_enc, 
						[None] + available_fields, [None] + self.fields, 1 + self.max_field_num)
					# x quantitative, no y field
					if y_enc_val == 0:
						y_agg_pout, y_agg_val = self.extend_by_choice(y_agg, [None], self.aggregates)
						x_topic_agg_pout, x_topic_agg_val = self.extend_by_choice(x_topic_agg, [None], self.aggregates)
					# x quantitative, y nominal
					elif self.translate_type(self.fields[y_enc_val - 1]) in ['nominal', 'temporal']:
						y_agg_pout, y_agg_val = self.extend_by_choice(y_agg, [None], self.aggregates)
						x_topic_agg_pout, x_topic_agg_val = self.extend_by_choice(x_topic_agg, [None], self.aggregates)
				if  self.marks[mark_val] in ['boxplot', 'text'] or y_enc_val == 0: ## the step function should also follow the rules
					color_enc_pout, color_enc_val = self.cut_branch(color_enc, [None] + self.fields, 1 + self.max_field_num)
					color_agg_pout, color_agg_val = self.cut_branch(color_agg, self.aggregates)
					# print(act_val, x_topic_enc_val, mark_val, y_enc_val, y_agg_val, x_topic_agg_val, color_enc_val, color_agg_val)
				else:
					#quantitative x and y
					available_fields = []
					for f in self.fields:
						if self.translate_type(f) in ['nominal'] and self.get_cardinality(f) <= 5:
							available_fields.append(f)
					color_enc_pout, color_enc_val = self.extend_by_choice(color_enc, 
						[None] + available_fields, [None] + self.fields, 1 + self.max_field_num)
					color_agg_pout, color_agg_val = self.cut_branch(color_agg, self.aggregates)
					
					# print(available_fields, act_val, x_topic_enc_val, mark_val, y_enc_val, y_agg_val, x_topic_agg_val, color_enc_val, color_agg_val)
					# if color_enc_val == 0:
						
					# elif self.translate_type(self.fields[color_enc_val - 1]) in ['quantitative']:
					# 	color_agg_pout, color_agg_val = self.extend_by_choice(color_agg, ['bin'], self.aggregates)
					# elif self.translate_type(self.fields[color_enc_val - 1]) in ['nominal']:
					# 	color_agg_pout, color_agg_val = self.extend_by_choice(color_agg, [None], self.aggregates)
		return [act_pout, x_topic_enc_pout, mark_pout, y_enc_pout, y_agg_pout, x_topic_agg_pout, color_enc_pout, color_agg_pout], \
				[act_val, x_topic_enc_val, mark_val, y_enc_val, y_agg_val, x_topic_agg_val, color_enc_val, color_agg_val]
		
		

	
	def translate_type(self, field):
		if field == None:
			return "invalid type"
		dtype = self.df[field].dtype
		if dtype in ['datetime64[ns]'] or \
				field.lower() in ['year', 'date', 'time']:
			return 'temporal'
		elif dtype in ['object', 'str', 'category', 'bool']:
			return 'nominal'
		else:
			return 'quantitative'






if __name__ == "__main__":
	# torch.zeros([1], dtype = torch.bool)
	# logits_or_qvalues = torch.randn((1,3))
	# head = CategoricalMasked(logits=logits_or_qvalues, mask=torch.tensor([[1,1,1]], dtype=torch.bool))
	# head2 = CategoricalMasked(logits=logits_or_qvalues)
	# res = torch.stack((head.logits, head2.logits))
	# print(head.mask)
	from vega_datasets import data
	car = data.cars()
	a = consSampling(car, 82)
	print(a.translate_type(None))
	# print(a.fields)
	# print(None in [1])




