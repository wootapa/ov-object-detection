# ov-object-detection

## Bygg
1. wget http://registrationcenter-download.intel.com/akdlm/irc_nas/15693/l_openvino_toolkit_p_2019.2.242.tgz
2. docker build -t ov-object-detection .

## Openvino sample
python3 /opt/intel/openvino/inference_engine/samples/python_samples/classification_sample/classification_sample.py \
-m /root/openvino_models/ir/FP32/classification/squeezenet/1.1/caffe/squeezenet1.1.xml \
--labels /root/openvino_models/ir/FP32/classification/squeezenet/1.1/caffe/squeezenet1.1.labels \
-i /opt/app/images/fulltsynlig.jpg \
-nt 5

## Openvino sample med CPU
python3 /opt/intel/openvino/inference_engine/samples/python_samples/classification_sample/classification_sample.py \
-m /root/tf-model/frozen_inference_graph.xml \
-i /root/images/fulltsynlig.jpg \
-nt 5 \
--cpu_extension /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so \
-d CPU

## Konvertera tf modell till ir
Vid större modeller krävs mycket minne. "faster_rcnn_nas_coco" krävde 8GB.

https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html

/opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
--input_model frozen_inference_graph.pb \
--tensorflow_object_detection_api_pipeline_config pipeline.config \
--tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json \
--reverse_input_channels \
--input_shape [1,600,600,3] \
--input image_tensor \
--output detection_scores,detection_boxes,num_detections \
--data_type FP16

## Printa tf version
python3 -c 'import tensorflow as tf; print(tf.__version__)'