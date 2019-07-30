#!/usr/bin/env python

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IEPlugin
from PIL import Image

model_xml = '/opt/app/model/frozen_inference_graph.xml'
model_bin = os.path.splitext(model_xml)[0] + ".bin"
lib_dir = None #Found in env #'/opt/intel/openvino/inference_engine/lib/intel64'
test_image = '/opt/app/images/fulltsynlig.jpg'
device = 'CPU'
cpu_lib = 'libcpu_extension_sse4.so'

# Load CPU plugin?
plugin = IEPlugin(device=device, plugin_dirs=lib_dir)
if device == 'CPU':
    plugin.add_cpu_extension(cpu_lib)


net = IENetwork(model=model_xml, weights=model_bin)
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))

# Openvino bugg, https://github.com/opencv/dldt/issues/128
# Ytterligare lösning, https://github.com/opencv/cvat/pull/545
iter = iter(net.inputs)
input_blob = None
while True:
    input_blob = next(iter, None)
    if input_blob is None or len(net.inputs[input_blob].shape) == 4:
        break

exec_net = plugin.load(network=net)


def pre_process_image(image_path):
    # Model input format
    n, c, h, w       = [1, 3, 600, 600] 
    image            = Image.open(image_path)
    processed_image  = image.resize((h, w), resample=Image.BILINEAR)
    
    # Normalize to keep data between 0 - 1
    processed_image  = (np.array(processed_image) - 0) / 255.0
    
    # Change data layout from HWC to CHW
    processed_image = processed_image.transpose((2, 0, 1))
    processed_image = processed_image.reshape((n, c, h, w))
    
    return processed_image

# Run inference
image = pre_process_image(test_image)
#res = exec_net.infer(inputs={: image})
res = exec_net.infer(inputs={input_blob: image})
res = res[out_blob]
print(res)

