import xml.etree.ElementTree as ET
import random
import numpy as np

def gen_xml_file_bco(init=None):
	"""Edits the template block_cmplx_obs.xml file to randomize block shape
	and saves new .xml as bco.xml.

	:param init: optional specification init of red obstacle
	"""
	tree = ET.parse('env/block_cmplx_obs.xml')  
	root = tree.getroot()
	# x,y = (x,y) pos of tip of red obstacle
	# t, m, b = length of top, middle, and back obstacle segments
	if init is not None:
		x,y,t,m,b=init
	else:
		#Pos of tip
		x,y = np.random.uniform(-1.5,1.5),np.random.uniform(-1.5,1.5)
		#Size of segments
		t, m, b = np.random.uniform(.75,1.5,size=3)
	# changing a field text
	for elem in root.iter('body'): 
		if elem.get("name") == 'torso':
			elem.set("pos", "%s %s .12" %(x,y))

			for elem2 in elem.iter('geom'):
				if elem2.get("name") == 't':
					elem2.set("fromto", "0 0 0 -%s 0 0" % t)
				if elem2.get("name") == 'm':
					elem2.set("fromto", "0 0 0 -%s 0 0" % m)
				if elem2.get("name") == 'b':
					elem2.set("fromto", "0 0 0 -%s 0 0" % b)
			for elem3 in elem.iter('body'):
				if elem3.get("name") == 'mid':
					elem3.set("pos", "-%s 0 0" % t)
				if elem3.get("name") == 'back':
					elem3.set("pos", "-%s 0 0" % m)
				if elem3.get("name") == 'bBall':
					elem3.set("pos", "-%s 0 =-.05" % b)
	tree.write("env/bco.xml")
	return [x,y,t,m,b]




def gen_xml_file_cb(bpos=None,test=0,size=.05,possible_units=range(4,10+1)):
	"""Edits the template cb.xml file to randomize block shape
	and saves new .xml as cb_new.xml.

	:param bpos: optional specification init of blue block
	:param test: if False, create any block made from 4-10 units; 
		if True, can only make blocks from 3 or11 units
	:param size: block side length / 2
	:param units: (min # blocks to use, max # blocks to use)
	:returns: # units used, positions of all units, (x,y) of last placed unit
	"""

	if test==2:
		possible_units=[3,11]
	total=sum([2**i for i in possible_units])
	#Favor producing blocks with higher 
	#number of units to promote diversity
	prob_units=[2**i/total for i in possible_units]
	num_blocks=np.random.choice(possible_units,p=prob_units)
	block_width = 2*size
	delta=np.array([[block_width,0],[-block_width,0],[0,block_width],[0,-block_width]])
	size = "%.2f %.2f %.2f" % (size, size, size)
	re,gr,bl = [0., .1, .6]
	rgba= "%.2f %.2f %.2f 1" % (re,gr,bl)

	#Randomly construct new blue block
	if bpos is None: 
		bpos = {'b1':[0,0],'b2':[0,-block_width]}
		ln=random.randint(1,2)
		last =np.array(bpos['b%d'%ln])
		for b in range(3,num_blocks+1):
			rn=random.randint(0,3)
			new_block_pos=last+delta[rn]
			while list(new_block_pos) in bpos.values():
				rn=random.randint(0,3)
				new_block_pos=last+delta[rn]
			bpos['b%d'%b]=list(new_block_pos)
			last=new_block_pos
	tree = ET.parse('./env/cb.xml')  
	root = tree.getroot()
	for main in root.iter('body'): 
		for elem in main.iter('geom'):
			name= elem.get("name")
			if name in bpos:
				elem.set("rgba",rgba)
				pos="%.2f %.2f .05" % (bpos[name][0],bpos[name][1])
				elem.set("pos", pos)
				elem.set("size",size)


		for j in main.iter("joint"):
			if j.get("name") == 'obj2_slidex':
				x,y = last
				j.set("pos", "%.2f %.2f 0.5" %(x,y))

	tree.write("./env/cb_new.xml")
	return num_blocks, bpos, last


def gen_walls():
	rgba= ".8 .8 .8 1"
	tree = ET.parse('env/cb_new.xml')  
	root = tree.getroot()
	for main in root.iter('body'): 
		for elem in main.iter('geom'):
			name= elem.get("name")
			if name == 'o1' or name == 'o2':
				elem.set("rgba",rgba)
	tree.write("env/cb_new.xml")

