import json
import numpy as np
from PIL import Image, ImageDraw,ImageFont
import cv2
import numpy as np
import os
import glob
import time

def clear_dir(directory):
	""" Remove all files from the directory """
	files = glob.glob(os.path.sep.join([directory, "*"]))
	for f in files:
		os.remove(f)

def splitter(file_name,frames_path,maxFrame):
	cap = cv2.VideoCapture(file_name)
	try:
		if not os.path.exists(frames_path):
			os.makedirs(frames_path)
	except OSError:
		print ('Error: Creating directory of data')
	clear_dir(frames_path)
	currentFrame = 0
	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		if not ret:
			break
		# Saves image of the current frame in png file
		name = frames_path + str(currentFrame) + '.png'
		cv2.imwrite(name, frame)

		# To stop duplicate images
		currentFrame += 1
		if(currentFrame>=maxFrame):
			print (str(currentFrame)+" frames created")
			break
	# When everything done, release the capture
	cap.release()
	return (currentFrame-1)

def check_clear(mask,W,H):
	for i in range(W):
		for j in range(H):
			if((mask[i][j])==False):
				return False
	return True
def one_frame_map(frames_path,one_frame_mask,colors,detections,W,H,f_num):
	for i in range(W):
		for j in range(H):
			one_frame_mask[i][j]=True
	for i in detections:
		x_min=i[0]
		x_max=i[2]
		if x_max<x_min:
			x_min=i[2]
			x_max=i[0]
		y_min=i[1]
		y_max=i[3]
		if y_max<y_min:
			y_min=i[3]
			y_max=i[1]
		if((y_max<H)and(x_max<W)):
			for j in range(x_max-x_min):
				for k in range(y_max-y_min):
					one_frame_mask[x_min+j][y_min+k]=False
	im = Image.open(frames_path+str(int(f_num))+".png")
	pix = im.load()
	for i in range(W):
		for j in range(H):
			if((one_frame_mask[i][j])==True):
				colors[i][j][0]+=1
				for k in range(3):
					colors[i][j][1][k]+=pix[i,j][k]
	return one_frame_mask

class Remover:
	def __init__(self,W,H,vid_path,data_path):
		self.W=W
		self.H=H
		self.data_path=data_path#path to removers data
		self.vid_path=vid_path#path to video
		self.maxFrame=1000
		self.frames_path=os.path.join(data_path,"frames/frame")
		self.frame_by_frame=[]
		with open(os.path.join(self.data_path,"frame_by_frame.txt"), 'r') as f:
			self.frame_by_frame = json.load(f)
		self.get_empty()
	def get_empty(self):
		size=splitter(self.vid_path,self.frames_path,self.maxFrame)
		mask=np.ndarray(shape=(self.W,self.H), dtype=bool)
		f_num=0
		colors=[]
		start=time.time()
		for i in range(self.W):
			colors.append([])
			for j in range(self.H):
				colors[i].append([0,[0,0,0]])
		for i in self.frame_by_frame:
			o_f=np.ndarray(shape=(self.W,self.H), dtype=bool)
			o_f=one_frame_map(self.frames_path,o_f,colors,i,self.W,self.H,f_num)
			#mask=np.logical_or(mask,o_f)
			if(((f_num%100)==0)and(f_num!=0)):
				print(f_num)
				print(round((time.time()-start),2))
				start=time.time()
			if((f_num)>=size):
				break
			#if(check_clear(mask,W,H))is True:
				#break
			f_num+=1
		im = Image.new('RGB', (self.W, self.H))
		pix = im.load()
		for i in range(self.W):
			for j in range(self.H):
				buf=[0,0,0]
				if(colors[i][j][0]!=0):
					for k in range(3):
						buf[k]=int(colors[i][j][1][k]/colors[i][j][0])
				pix[i,j]=tuple(buf)
		im.save(os.path.join(self.data_path,"empty_frame.png"))