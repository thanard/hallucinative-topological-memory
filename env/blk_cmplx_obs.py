"""
Complex Obstacle Environment.

A green block on a table with a red obstacle that can 
vary in shape, mass, and position.
The agent can apply a small displacement to the green block.
"""

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs
import random
import scipy.misc
import time
import argparse
from env.edit import gen_xml_file_bco
import os

class BlkCmplxObs(MujocoEnv, Serializable):
	dir_path = os.path.dirname(os.path.realpath(__file__))
	FILE = os.path.join(dir_path, '/bco.xml')

	def __init__(self, *args, **kwargs):
		super(BlkCmplxObs, self).__init__(*args, **kwargs)
		Serializable.quick_init(self, locals())

	def get_current_obs(self):
		return self.model.data.geom_xpos

	def get_current_img(self, config=None):
		for _ in range(20):
			img = self.render(mode='rgb_array', config=config)
		img = scipy.misc.imresize(img, (64, 64, 3), interp='nearest')
		return img

	def viewer_setup(self):
		self.viewer.cam.trackbodyid = 0
		self.viewer.cam.distance = 7
		self.viewer.cam.elevation = -90


	def step(self,action,joints,red_first):
		"""Take a single step.

		:param action: action to take
		:param joints: joint hparams of the red obstacle
		:param red_first: sum of red pixels in the context image
		"""
		block_pos = self.get_current_obs()[8][:2]
		self.reset(block_pos+action,joints=joints)
		for _ in range(20):
			img=self.render(config=config,mode='rgb_array')
		img= scipy.misc.imresize(img, (64, 64, 3), interp='nearest')
		red_curr = (img[:,:,0]/255*(img[:,:,0]/255>.8)).sum()

		#Ensure the step doesn't force the block into the obstacle
		while abs(red_curr-red_first)>.5:
			self.reset(block_pos+np.random.uniform(-0.05,0.05,size=2),joints=joints)
			for _ in range(20):
				img=self.render(config=config,mode='rgb_array') 
			img= scipy.misc.imresize(img, (64, 64, 3), interp='nearest')
			red_curr = (img[:,:,0]/255*(img[:,:,0]/255>.8)).sum()

		return img

	def reset(self, block_pos=None, joints=None,context_only=False):
		"""Reset block. 

		:param block_pos: optional (x,y) of block to set to
		:param joints: optional joints to set obstacle to; else randomizes joints
		:param context_only: flag if context only; removes block from table
		"""


		## init_pos[7:9]= [x,y] of block 
		## self.model.body_pos:
			# 0 - table
			# 1 - rot0
			# 2 - torso-tip joint
			# 3 - rot1
			# 4 - torso-mid ball joint
			# 5 - mid-back joint
			# 6 - rot3
			# 7 - back-tip joint
			# 8 - block
			# 9..12 - walls 

		init_pos = np.zeros((9))
		if block_pos is None:    
			block_pos = np.random.uniform(-2,2,size=(2)) - self.model.body_pos[8, :2]
		elif not context_only:
			block_pos = np.clip(block_pos,-2.1,2.1)
			block_pos -= self.model.body_pos[8, :2]
		else:
			block_pos -= self.model.body_pos[8, :2]
		init_pos[7:9] = block_pos

		if joints is None:
			joints = np.random.uniform(-1.5,1.5,size=3)
		init_pos[0:2] = np.array([joints[0],0])-self.model.body_pos[2,:2]
		init_pos[2:4] = np.array([joints[1],0])-self.model.body_pos[4,:2]
		init_pos[4:6] = np.array([joints[2],0])-self.model.body_pos[5,:2]

		full_state = np.concatenate([init_pos, self.init_qvel.squeeze(),
					  self.init_qacc.squeeze(), self.init_ctrl.squeeze()])
		obs = super(BlkCmplxObs, self).reset(init_state=full_state)

		return obs, joints

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Move some blocks')
	parser.add_argument('--length', type=int, default=50, help='max # steps after which you terminate a single simulation')
	parser.add_argument('--n_trials', type=int, default=10, help='number of simulations')
	parser.add_argument('--n_contexts', type=int, default=10)
	parser.add_argument('--visible', action='store_true', help='flag if do not want headless rendering')
	parser.add_argument('--test', action='store_true', help='flag if creating test data')


	args = parser.parse_args()
	config = {
		"visible": args.visible,
		"init_width": 128,
		"init_height": 128,
		"go_fast": True
	}
	start_time = time.time()
	all_data = []
	n_trials = args.n_trials
	n_contexts = args.n_contexts

	for n in range(n_contexts):
		first = True
		print("##### Context %d/%d ######" % (n, n_contexts))
		init=gen_xml_file_bco()

		#Find valid obstacle position
		while first or np.sum(c)==0:
			first = False
			env = BlkCmplxObs()
			_,joints=env.reset(block_pos=np.array([4.,4.]),context_only=True)
			env.render(config=config,mode='rgb_array')
			env.viewer_setup()
			for _ in range(20):
				c = env.render(config=config,mode='rgb_array')
			c = scipy.misc.imresize(c, (64, 64, 3), interp='nearest')
			c_pos = env.get_current_obs()[:8]
			if c_pos.reshape(-1).max() > 2.1 or c_pos.reshape(-1).min() < -2.1:
				first = True
				env.stop_viewer()
		red_first = (c[:,:,0]/255*(c[:,:,0]/255>.8)).sum()

		for it in range(n_trials):
			first2 = True
			data=[]      
			while first2:
				first2=False
				env.reset(joints=joints)
				env.render(config=config,mode='rgb_array')
				for _ in range(20):
					img=env.render(config=config,mode='rgb_array')
				img = scipy.misc.imresize(img, (64, 64, 3), interp='nearest')
				block_pos = env.get_current_obs()[8][:2]
				for obs in env.get_current_obs()[:8]:
					if sum((block_pos - obs[:2])**2) < .20:
						first2 = True
			for t in range(args.length):

				delta = np.random.uniform(-.1,.1,size=2)
				block_pos = env.get_current_obs()[8][:2]
				env.reset(block_pos+delta,joints=joints)
				for _ in range(20):
					img=env.render(config=config,mode='rgb_array')
				img= scipy.misc.imresize(img, (64, 64, 3), interp='nearest')
				if t == 0: 
					start = img
					block_pos_s=env.get_current_obs()[8][:2]

				red_curr = (img[:,:,0]/255*(img[:,:,0]/255>.8)).sum()

				if abs(red_curr-red_first)>.5: 
					env.reset(block_pos-.5*delta,joints=joints)
					for _ in range(20):
						img=env.render(config=config,mode='rgb_array') 
					img= scipy.misc.imresize(img, (64, 64, 3), interp='nearest')
					delta *= -.5

				label = {'action':delta,'state':env.get_current_obs()[:9]}
				data.append((np.concatenate((img,c),axis=2),label))

			if args.test:
				block_pos_g = env.get_current_obs()[8][:2]
				label = [list(block_pos_s),list(block_pos_g),list(joints),init]  
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
		fname = 'data/test_bco_context_%d' % n_contexts
	else:
		fname = 'data/train_bco_traj_length_%d_n_trials_%d_context_%d.npy' % (
			args.length, n_trials, n_contexts)
	np.save(fname, all_data)

