import interface as bbox
import numpy as np
from sklearn.externals import joblib

lr_coefs_1 = joblib.load('lr_coefs_1')
lr_free_coefs_1 = joblib.load('lr_free_coefs_1')

def run_bbox():
	f_35_penalty = 0.15; k = 0
	bbox.load_level("levels/test_level.data", verbose=0)
	has_next = True; last_score = 0
	act = -1; act_len = 0; crit_len = 150
	predict = np.zeros(2); cum_sum = np.zeros(4)
	while has_next:
		last_act = act
		state = bbox.get_state()
		predict[:2] = np.dot(lr_coefs_1,state[:-1]) + lr_free_coefs_1

		if state[35] > 0:
			cum_sum[1] = predict[0] + k
			cum_sum[2] = -predict[0] + k
		elif state[35] < 0:
			cum_sum[1] = -predict[1] + k
			cum_sum[2] = predict[1] + k
		elif state[35] == 0:
			cum_sum[1] = predict[0] + k
			cum_sum[2] = predict[1] + k

		cum_sum[0] = (cum_sum[1]+cum_sum[2])/2 + k
		cum_sum[1]-=f_35_penalty*state[35]
		cum_sum[2]+=f_35_penalty*state[35]
		if act_len > crit_len: cum_sum[last_act]-=0.0078125
		act = cum_sum.argmax()

		has_next = bbox.do_action(act)
		if last_act==act: act_len+=1
		else: act_len = 0

	bbox.finish(verbose=1)

run_bbox()
