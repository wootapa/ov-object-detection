#!/usr/bin/env python
# coding: utf-8
import base64
import json
import math
import os
import subprocess
from io import BytesIO

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from werkzeug import secure_filename

import cv2
from flask import Flask, jsonify, request
from openvino.inference_engine import IENetwork, IEPlugin

model_xml = '/opt/model/frozen_inference_graph.xml'
model_bin = '/opt/model/frozen_inference_graph.bin'
labels_path = '/opt/model/mscoco_label_map.txt'
lib_dir = None #Found in env #'/opt/intel/openvino/inference_engine/lib/intel64'
device = 'CPU'
cpu_lib = 'libcpu_extension_sse4.so'
result_count = 10
result_threshold = 0.5
bad_labels = ['person', 'car', 'truck', 'motorcycle']

class ModelManager():
    def __init__(self):
        self._model = model_xml
        self._weights = model_bin
        self._labels_map = {}
        
        # Load labels
        with open(labels_path, 'r') as f:
            for line in f:
                (key, val) = line.split(':')
                self._labels_map[int(key)] = val.rstrip().lower()

        # Load CPU plugin?
        plugin = IEPlugin(device=device, plugin_dirs=lib_dir)
        if device == 'CPU':
            plugin.add_cpu_extension(cpu_lib)
        
        network = IENetwork(model=self._model, weights=self._weights)

        supported_layers = plugin.get_supported_layers(network)
        not_supported_layers = [l for l in network.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            raise Exception("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ", ".join(not_supported_layers)))

        iter_inputs = iter(network.inputs)
        self._input_blob_name = next(iter_inputs)
        self._output_blob_name = next(iter(network.outputs))

        self._require_image_info = False

        # NOTE: handeling for the inclusion of `image_info` in OpenVino2019
        if 'image_info' in network.inputs:
            self._require_image_info = True
        if self._input_blob_name == 'image_info':
            self._input_blob_name = next(iter_inputs)

        self._net = plugin.load(network=network, num_requests=2)
        input_type = network.inputs[self._input_blob_name]
        self._input_layout = input_type if isinstance(input_type, list) else input_type.shape

    def infer(self, image):
        _, _, h, w = self._input_layout
        in_frame = image if image.shape[:-1] == (h, w) else cv2.resize(image, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        inputs = {self._input_blob_name: in_frame}
        if self._require_image_info:
            info = np.zeros([1, 3])
            info[0, 0] = h
            info[0, 1] = w
            # frame number
            info[0, 2] = 1
            inputs['image_info'] = info

        infer_results = self._net.infer(inputs)
        infer_result = infer_results[self._output_blob_name]

        # We're done. Make view objects.
        predictions = [pred for pred in infer_result[0][0] if pred[2] > result_threshold]
        result_list = []
        for pred in predictions:
            class_id = int(pred[1])
            category = self._labels_map[class_id] if self._labels_map else "{}".format(class_id)
            score = math.ceil(pred[2] * 100)
            result_list.append(Result(category, score))
        
        result_image = self.plot_result_image(image, predictions)
        result_summary = ResultSummary(result_list, result_image)
        return result_summary
    
    def plot_result_image(self, image, predictions):
        plt.clf()
        ax = plt.subplot(1, 1, 1)
        ax.set_frame_on(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.autoscale(tight=True)
        plt.axis('off')
        plt.imshow(image)  # slice by z axis of the box - box[0].

        for pred in predictions:
            class_label_id = int(pred[1])
            class_label = self._labels_map[class_label_id] if self._labels_map else "{}".format(class_label_id)
            score = math.ceil(pred[2] * 100)
            color = 'red' if class_label in bad_labels else 'blue'
            box = pred[3:]
            box = (box * np.array(image.shape[:2][::-1] * 2)).astype(int)
            x_1, y_1, x_2, y_2 = box
            x_1 -= 20
            x_2 += 20
            y_1 -= 20
            y_2 += 20
            rect = patches.Rectangle((x_1, y_1), x_2-x_1, y_2 - y_1, linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x_1 + 2, y_1 - 10, '{} {}%'.format(class_label, score), fontsize=6, color='white', bbox=dict(facecolor=color, edgecolor='none', pad=1.0))
        
        buf = BytesIO()
        plt.savefig(buf, bbox_inches = 'tight', pad_inches = 0, format="jpg")
        return base64.b64encode(buf.getbuffer()).decode("ascii")


class Result:
    def __init__(self, category, score):
        self.category = category
        self.score = score

    def toJson(self):  
        return {           
            'category': self.category, 
            'score': self.score
        }

class ResultSummary:
    def __init__(self, results, imagebase64):
        self.results = results
        self.image = ("data:image/jpeg;base64," + imagebase64)
    
    def toJson(self):  
        return {           
            'results': [result.toJson() for result in self.results], 
            'image': self.image
        }

mgr = ModelManager()
app = Flask(__name__, static_folder=os.path.dirname(os.path.realpath(__file__)))

@app.route('/')
def index():
   return app.send_static_file('index.html')

@app.route('/categories', methods=['GET'])
def categories():
    categories = [] 
    for key in mgr._labels_map:
        categories.append(mgr._labels_map[key])
    categories.sort()
    return jsonify(categories)

@app.route('/classify', methods=['POST'])
def classify():
    img = Image.open(request.files['file'])
    
    # Maybe convert png->jpeg
    if not img.mode == 'RGB':
        img = img.convert('RGB')
    
    # Infer...
    summary = mgr.infer(np.array(img))
    return jsonify({'summary':summary.toJson()})

if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(port=5000, host=('0.0.0.0'))
