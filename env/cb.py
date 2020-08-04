"""
Complex Block Environment.

A blue block that can vary in shape and mass is on a table.
The agent can apply a small displacement to the blue block, 
and can also rotate it from a single affector. 
"""

import os
import numpy as np
import scipy.misc
import time
import argparse
import random

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs
from env.edit import gen_xml_file_cb, gen_walls

class ComplexBlock(MujocoEnv, Serializable):
	global FILE

	def __init__(self,units_used, *args, **kwargs):
		"""
		:units_used: denotes the number of units that 
		make up the larger blue block
		"""
		self.units_used=units_used
		dir_path = os.path.dirname(os.path.realpath(__file__))
		FILE=os.path.join(dir_path, '/cb_new.xml')
		super(ComplexBlock, self).__init__(*args, **kwargs, file_path=FILE)
		Serializable.quick_init(self, locals())

	def get_current_obs(self):
		return self.model.data.geom_xpos.copy()

	def get_current_img(self, config=None):
		img = self.render(mode='rgb_array', config=config)
		img = scipy.misc.imresize(img, (64, 64, 3), interp='nearest')
		return img

	def viewer_setup(self):
		self.viewer.cam.trackbodyid = 0
		self.viewer.cam.distance = 2.
		self.viewer.cam.elevation = -90

	def step(self, action):
		self.forward_dynamics(action)
		next_obs = self.get_current_obs()
		return Step(next_obs, 0, False)

	def illegal_init(self):
		"""Detect if there is any contact. 
		"""
		b_pos=self.get_current_obs()[1:self.units_used+1,:2]
		if abs(b_pos).max() > .8:
			return True
		for (x,y) in b_pos:
			if abs(x) != 0 and abs(y) > .15 and abs(x) < .12:
				return True
		return False
		

	def reset(self, b_pos=None,init=False):
		"""Resets the blue block on the table.

		:param b_pos: optional param to specify location of block
		:init: flag as True if initializing a new block shape / context
		"""
		init_state = np.zeros((3,3))
		if b_pos is None:
			b_pos=np.random.uniform(-.8,.8,size=2)
		if init:
			b_pos = np.zeros(2)
		init_state[1, :2] = b_pos
		init_pos = (init_state)[1:, :].reshape(-1)
		full_state = np.zeros((1, 2*3*4)) 
		full_state[0, :3*2] = init_pos 
		super(ComplexBlock, self).reset(init_state=full_state)
		return self.get_current_obs()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Move some blocks')
	parser.add_argument('--length', type=int, default=50, help='max # steps after which you terminate a single simulation')
	parser.add_argument('--n_trials', type=int, default=10, help='number of simulations')
	parser.add_argument('--n_contexts', type=int, default=10)
	parser.add_argument('--visible', action='store_true', help='flag if do not want headless rendering')
	parser.add_argument('--test', action='store_true', help='flag if creating test data, only produces \
		blue blocks made of 3 or 11 units')
	parser.add_argument('--env_info', action='store_true', help='flag if store env info, needed to resimulate ie. for testing')


	args = parser.parse_args()
	config = {
		"visible": args.visible,
		"init_width": 128,
		"init_height": 128,
		"go_fast": True
	}
	start_time = time.time()
	all_data = []
	test=args.test
	n_trials = args.n_trials
	n_contexts = args.n_contexts

	for e in range(n_contexts):
		units_used, bpos, last = gen_xml_file_cb(test=test)
		env = ComplexBlock(units_used=units_used)

		print("##### Context %d/%d ######" % (e, n_contexts))
		env.reset(init=True)
		env.render(mode="rgb_array",config=config)
		env.viewer_setup()
		for _ in range(20):
			c=env.render(mode="rgb_array",config=config)
		while c.sum() == 0:
			c = env.render(mode='rgb_array',config=config)
		c = scipy.misc.imresize(c, (64, 64, 3), interp='nearest')

		gen_walls()
		env = ComplexBlock(units_used=units_used)
		env.reset(b_pos=np.array([.6,.1]))
		env.render(mode="rgb_array",config=config)
		env.viewer_setup()
		for _ in range(20):
			env.render(mode="rgb_array",config=config)


		for i in range(n_trials):
			env.reset()
			while env.illegal_init():
				env.reset()

			img = env.render(mode='rgb_array', config=config)
			img = scipy.misc.imresize(img, (64, 64, 3), interp='nearest')
			start_state=list(env.model.data.qpos.reshape(-1)[:2])

			data=[]
			for t in range(args.length):
				action=np.random.uniform(-.5,.5,size=4)
				env.step(action)
				for _ in range(50):
					img = env.render(mode='rgb_array', config=config)
					env.step(np.zeros(4))

				img = scipy.misc.imresize(img, (64, 64, 3), interp='nearest')
				if t == 0:
					start = img

				label = {'action': action, 'units':len(bpos)}
				data.append((np.concatenate((img,c),axis=2),label))
			goal_state=list(env.model.data.qpos.reshape(-1)[:2])
			if args.test:
				label=[units_used,bpos,list(last),start_state,goal_state]
				all_data.append((np.concatenate((start[None,:],img[None,:],c[None,:]),axis=0),label))
			else:
				all_data.append(data)

		env.stop_viewer()

	total = time.time() - start_time
	print("number of trajectories: %d" % (len(all_data)))
	print("number of samples: %d" % (args.length*len(all_data)))
	print("took %d seconds"%total)
	if not os.path.exists('data'):
		os.mkdir('data')
	if args.test:
		fname = 'data/test_cb_context_%d' % n_contexts
	else:
		fname = 'data/train_cb_traj_length_%d_n_trials_%d_context_%d.npy' % (
			args.length, n_trials, n_contexts)
	np.save(fname, all_data)
