import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import cv2
import time

curdir = os.getcwd()  # trick for good import
sys.path.append(curdir + '/tf_models/research')  # <path to tensorflow models repo>/research
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
os.chdir(curdir)  # end trick

# Define the video stream
cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams

# What model to download.
# Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = MODEL_NAME + '/label_map.pbtxt'

# Number of classes to detect
NUM_CLASSES = 1

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Detection
with detection_graph.as_default():
    # Extract image tensor
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Extract detection boxes
    boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Extract detection scores
    scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
    # Extract detection classes
    classes_tensor = detection_graph.get_tensor_by_name('detection_classes:0')
    # Extract number of detectionsd
    num_detections_tensor = detection_graph.get_tensor_by_name('num_detections:0')

    with tf.Session(graph=detection_graph) as sess:
        while True:
            t1 = time.time()
            # Read frame from camera
            ret, image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes_tensor, scores_tensor, classes_tensor, num_detections_tensor],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            t = time.time() - t1
            # Display output
            print('getting photo and object detection on GPU takes {0} s'.format(t))
            cv2.imshow('', cv2.resize(image_np, (800, 600)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break