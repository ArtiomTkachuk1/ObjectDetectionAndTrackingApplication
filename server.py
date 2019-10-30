import os;
from werkzeug.utils import secure_filename;
# from flask_sqlalchemy import SQLAlchemy;
import json;
import cv2;
import youtube_dl;
from collections import Iterable;
import imutils;
import time;
import glob;
import math as m;
from flask import Flask, render_template, request, redirect, url_for, request, send_file, Response,abort;
from modules.traffic_counter.streamer import *;
from modules.Viz.vizualizator import *;
from modules.Ob_Rem.objects_remover import *;
from modules.G_Prop.gate_proposer import *;

app = Flask(__name__, static_folder="build/static", template_folder="build")

UPLOAD_FOLDER = os.path.join(app.root_path, 'upload')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
DATA_FOLDER = os.path.join(app.root_path, 'data')
app.config['DATA_FOLDER'] = DATA_FOLDER

frame_name = "frame.jpg"
link_name = "link.txt"
lines_name="settings.json"
setting_name="appsettings.json"
current_det_name="current_det.txt"

def check_youtube(src, quality="opt"):
	if src[:24] == "https://www.youtube.com/" or src[:17] == "https://youtu.be/":
		ydl_opts = {}
		ydl = youtube_dl.YoutubeDL(ydl_opts)
		info_dict = ydl.extract_info(src, download=False)
		formats = info_dict.get('formats', None)
		qualities = dict([(i, m.sqrt(f.get('height') ** 2 + f.get('width') ** 2)) for i, f in enumerate(formats)
						  if f.get('height') is not None and f.get('width') is not None])
		index = -1
		max_int32 = 2147483647
		if quality == "min":
			min_diag = max_int32
			for idx, diag in qualities.items():
				if diag < min_diag:
					min_diag = diag
					index = idx
		elif quality == "max":
			max_diag = -1
			for idx, diag in qualities.items():
				if diag > max_diag:
					max_diag = diag
					index = idx
		'''elif quality == "opt":
			min_diff = max_int32
			net_diag = m.sqrt(net_input_size[0] ** 2 + net_input_size[1] ** 2)
			for idx, diag in qualities.items():
				diff = diag - net_diag
				diff = diff if diff >= 0 else max_int32
				if diff < min_diff:
					min_diff = diff
					index = idx'''
		src = formats[index].get('url', None)
	return src


imgW=0
imgH=0
def first_frame_getter(vid_path):
	global imgW
	global imgH
	global frame_name
	frame_path=os.path.join(app.config['DATA_FOLDER'],frame_name)
	vs = cv2.VideoCapture(vid_path);
	grabbed, frame = vs.read();
	fourcc = cv2.VideoWriter_fourcc(*"MJPG");
	writer = cv2.VideoWriter(frame_path, fourcc, 30, (frame.shape[1], frame.shape[0]), True);
	writer.write(frame);
	imgW=frame.shape[1]
	imgH=frame.shape[0]


@app.route('/')
def index():
	return render_template('index.html')

settings_App={}
@app.route('/', methods=['POST'])
def upload_file():
	global settings_App
	global imgW
	global imgH
	global link_name
	global lines_name
	global setting_name
	link_path=os.path.join(app.config['DATA_FOLDER'],link_name)
	lines_path=os.path.join(app.config['DATA_FOLDER'],lines_name)
	setting_path=os.path.join(app.config['DATA_FOLDER'],setting_name)
	if request.method == 'POST':
		type="";
		if(isinstance(request.json, Iterable)):
			for i in request.json:
				if((i=="ref")or(i=="lines")):
					type=i;
					break
			for i in request.json:
				if(i=="nn"):
					settings_App["nn"]=request.json[i];
					print(request.json[i])
		elif(isinstance(request.files, Iterable)):
				for i in request.files:
					if i=="video":
						type=i;
		vid_path=""
		if((type=='ref')or(type=='video')):
			if(type=='video'):
				file = request.files[type];
				filename=file.filename;
				vid_path=os.path.join(app.config['UPLOAD_FOLDER'],filename);
				file.save(vid_path);
				cap = cv2.VideoCapture(vid_path)
				settings_App["num_of_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
				with open(setting_path, 'w') as json_file:
					json.dump(settings_App, json_file)
			if(type=="ref"):
				settings_App={}
				ref = request.json[type];
				vid_path=ref;
				vid_path = check_youtube(vid_path);
				settings_App["nn"]=request.json["nn"];
				settings_App["num_of_frames"]=int(request.json["num_of_frames"]);
				with open(setting_path, 'w') as json_file:
					json.dump(settings_App, json_file)
			file1 = open(link_path,"w")
			file1.write(vid_path)
			file1.close()
			first_frame_getter(vid_path);
		if(type=='lines'):
			lines_raw=json.loads(request.json[type])
			lines=[]
			for i in lines_raw:
				line=[]
				line.append([int(i['x0']*imgW/800),int(i['y0']*imgH/450)])
				line.append([int(i['x1']*imgW/800),int(i['y1']*imgH/450)])
				lines.append(line)
			temp_dict = {'lines': lines, "filter": list([[]] * len(lines))}
			json_dict = json.dumps(temp_dict)
			with open(lines_path, 'w') as json_file:
				json.dump(json_dict, json_file)
	return str(request)


@app.route('/get_image', methods=['GET'])
def return_image():
	global frame_name
	frame_path = os.path.join(app.config['DATA_FOLDER'], frame_name)
	return send_file(frame_path, attachment_filename=frame_name, cache_timeout=0)


def gen(Streamer, frame_max, frame_index):
	while frame_index < frame_max:
		frame = Streamer.get_frame()
		if frame!=None:
			frame_index = frame_index + 1
			yield (b'--frame\r\n'
				   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/get_current_det',methods=['GET'])
def return_data():
	global current_det_name
	global lines_name
	current_det_path=os.path.join(app.config['DATA_FOLDER'], current_det_name)
	lines_path = os.path.join(app.config['DATA_FOLDER'], lines_name)
	if os.path.exists(current_det_path):
		file = open(current_det_path,"r");
		data = file.read();
		file.close();
		if(json.loads(data)["exit"]==True):
			os.remove(current_det_path)
			os.remove(lines_path)
		return data
	else:
		return "{}"

@app.route('/get_stream', methods=['GET'])
def return_stream():
	global link_name
	global lines_name
	global setting_name
	link_path=os.path.join(app.config['DATA_FOLDER'],link_name)
	lines_path=os.path.join(app.config['DATA_FOLDER'],lines_name)
	setting_path=os.path.join(app.config['DATA_FOLDER'],setting_name)
	file1 = open(link_path, "r")
	vid_path = file1.readline()
	file1.close()
	with open(setting_path) as file:
		data = json.load(file)
	nn_type=data["nn"]
	frame_max=data["num_of_frames"]
	frame_index=0
	return Response(gen(Streamer(vid_path,lines_path,nn_type,frame_max,app.config['DATA_FOLDER']),frame_max,frame_index),
					mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_stats', methods=['POST'])
def count_stats():
	camstats1=camstats(app.config['DATA_FOLDER'])
	return str(request)

@app.route('/get_stats0', methods=['GET'])
def return_stats0():
	im_path=os.path.join(app.config['DATA_FOLDER'],"detections.png")
	return send_file(im_path, attachment_filename=frame_name, cache_timeout=0)

@app.route('/get_stats1', methods=['GET'])
def return_stats1():
	im_path=os.path.join(app.config['DATA_FOLDER'],"frame_with_ways_lines.png")
	return send_file(im_path, attachment_filename=frame_name, cache_timeout=0)

@app.route('/get_stats2', methods=['GET'])
def return_stats2():
	im_path=os.path.join(app.config['DATA_FOLDER'],"frame_with_ways_lines_clear_smooth.png")
	return send_file(im_path, attachment_filename=frame_name, cache_timeout=0)

@app.route('/get_stats3', methods=['GET'])
def return_stats3():
	im_path=os.path.join(app.config['DATA_FOLDER'],"heatmap_avg_time.png")
	return send_file(im_path, attachment_filename=frame_name, cache_timeout=0)

@app.route('/get_stats4', methods=['GET'])
def return_stats4():
	im_path=os.path.join(app.config['DATA_FOLDER'],"heatmap_lengths.png")
	return send_file(im_path, attachment_filename=frame_name, cache_timeout=0)

@app.route('/get_empty_frame', methods=['POST'])
def count_empty():
	global imgW
	global imgH
	global link_name
	link_path=os.path.join(app.config['DATA_FOLDER'],link_name)
	print(link_path)
	file1 = open(link_path, "r")
	vid_path = file1.readline()
	file1.close()
	print(vid_path)
	remover1=Remover(imgW,imgH,vid_path,app.config['DATA_FOLDER'])
	return str(request)

@app.route('/get_empty', methods=['GET'])
def return_empty():
	im_path=os.path.join(app.config['DATA_FOLDER'],"empty_frame.png")
	return send_file(im_path, attachment_filename=frame_name, cache_timeout=0)

@app.route('/get_rec_gates', methods=['POST'])
def count_rec():
	ways_clear1=Ways_clear(app.config['DATA_FOLDER'])
	ways_clear1.clustering_meanshift()
	return str(request)

@app.route('/get_rec', methods=['GET'])
def return_rect():
	im_path=os.path.join(app.config['DATA_FOLDER'],"clustering_meanshift_lines.png")
	return send_file(im_path, attachment_filename=frame_name, cache_timeout=0)

print(app.url_map)
print('Starting Flask!')
if __name__ == '__main__':
	app.debug = True
	app.run(host='0.0.0.0')


'''
TO DO
1.secure post requests from injections,xss,etc

ALLOWED_EXTENSIONS = set(['avi', 'mp4'])
def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

'''
