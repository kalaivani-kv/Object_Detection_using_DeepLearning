#Object Detection In Live Streaming Videos
#Here the pretrained model for object detection in Live Streaming Video is used.

import numpy as np
import os 
import six.moves.urllib as urllib
import sys
import tarfile
import os 
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 0 = all messages are logged (deafult behavior)
# 1 = INFO messages are not printed 
# 2 = INFO and WARNING messages are not printed 
# 3 = INFO, WARNING and ERROR messages are not printed

import tensorflow as tf 
import gin.tf
import tensorflow.python.framework.dtypes
import zipfile

from distutils.version import StrictVersion 
from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt
from PIL import Image 

#Imports Libraries

# Select the camera you are using for live streaming 
cap = cv2.VideoCapture(0)
# Here we require path since the notebook is stored in the object_detection folder 
sys.path.append('C:/ProgramData/TensorFlow/models/research/object_detection')
from object_detection.utils import ops as utils_ops
# To display the images
%matplotlib inline
# Imports for object detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#Prepare Model
#We are using the pretrained model;you can export any model export_inference_graph.py tool ; just need to point the new.pb file

# The model we need to download 
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
MODEL = 'ssd_mobilenet_v1_coco_2017_11_17'
FILE = MODEL + '.tar.gz'
DOWNLOAD_URL = 'https://download.tensorflow.org/models/object_detection/'

# Actual model used for object detection and Path for frozen detection graph.
PATH_TO_FROZEN_DETECTION_GRAPH = MODEL + '/frozen_inference_graph.pb'

# Label strings which used for each box.
PATH_TO_BOX_LABELS = os.path.join('C:/ProgramData/TensorFlow/models/research/object_detection/data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
# Download the model 

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_URL + FILE, FILE)
tar_file = tarfile.open(FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
        
        
# Load the tensorflow model into memory 
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  if not tf.io.gfile.exists(PATH_TO_BOX_LABELS):
      tf.io.gfile.makedirs(PATH_TO_BOX_LABELS)
  with tf.io.gfile.GFile(PATH_TO_FROZEN_DETECTION_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name= '')


#Labels Map Loading

#category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_BOX_LABELS, use_display_name=True)
label_map = label_map_util.load_labelmap(PATH_TO_BOX_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# utility function 
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

#Object Detection

# we are using here 4 images and also use one direct download and use it img1.jpg,img2.jpg,img3.jpg,img4.ipg
TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(TEST_IMAGES_DIR,'img{}.jpg'.format(i)) for i in range(1, 6)]

# Size for images in inches direct output.
IMAGE_SIZE = (12, 8)

# inferences for single image
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      #get handles to input and output tensors 
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        #processing for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        #we need to translate mask from box coordinates to image coordinates and get fit the image size.
        real_num_detection =  tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Adding the back batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            
      #executing the inference
      output_dict = sess.run(tensor_dict,
                             feed_dict ={image_tensor: image})
            
      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

#Output for Live Streaming Video
# for the video and live streaming
while True:
  ret, image_np = cap.read()
  #expand dimensions model expecting the sample in format: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  #real object detection 
  output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
  #showing the boxes
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
  cv2.imshow('object detection', cv2.resize(image_np, (1200,800)))
  if cv2.waitKey(25) & 0xFF == ord('q'):
    cv2.destroyAllWindow()
    break