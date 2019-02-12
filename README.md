# TensorFlow Object Detection API Hangs On — Training and Evaluating using Custom Object Detector

## First of All

1. Install TensorFlow (I am using TensorFlow CPU this time. In the next post, I will explain how to install TensorFlow GPU)

 > [https://www.tensorflow.org/install/][1]

2. Clone the TensorFlow object detection repository

 > [https://github.com/tensorflow/models][2]

3. Install the Object Detection API package.

4. Motivation
 Currently, I am in charge of an ad-hoc project that used machine vision to detect whether a person is wearing safety goggles. I researched online and tried myself, and it goes well. Then I am thinking of sharing my learning and obstacles as not many people talk about how to deploy the custom trained model. In this post, I will explain all the necessary steps to train your own detector. The process of doing it is shown below:
 ![Object Detection Process][3]

45. Project Structure
 - images/ — our dataset of images.
 - labels/ - labels for our dataset.
 - data/ — records and .csv files.
 - train/ — our trained model.
 - eval/ — evaluation results of trained model.
 - output/ - inference graph
 - App/ - deployment of application.


## Creating Dataset

You need to prepare images as many as you can for training, but at least need to be more than 5 images per frame. Then hand-labeled them manually with [LabelImg][4]. LabelImg is a graphical image annotation tool that is written in Pyandn and uses Qt for the graphical interface. It’s super easy to use and the annotations are saved as XML files in the PASCAL VOC format to be used by the [generate_tfrecord.py][5] script.

After labeling the images, use the [xml_to_csv.py][6] script that converts the XML files to a .csv file and then created the TFRecords. I used 80/20 rule for training and testing.

Tensorflow Object Detection API uses the TFRecord file format, you need to convert our dataset to this file format. There are several options to generate the TFRecord files. Either you have a dataset that has a similar structure to the PASCAL VOC dataset or the Oxford Pet dataset, then they have ready-made scripts for this case (see create_pascal_tf_record.py and create_pet_tfd_record.py). If you don’t have one of those structures you need to write your own script to generate the TFRecords. I used a custom made script for this!

After labeling the images using LabelImg, labeled xml files will be generated. Run the [xml_to_csv.py][7], record down the number of Test Cases printed out in the console. Then generate TF Records for both training and testing using [generate_tfrecord.py][8].

To generate train.record file use the code as shown below:
```
python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record --image_dir=images
```
To generate test.record file use the code as shown below:
```
python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record --image_dir=images
```


## Training the model

Once our records files are ready, you are almost ready to train the model.

Firstly, you need to decide the pre-trained model to be used. There’s a tradeoff between detection speed and accuracy, higher the speed lower the accuracy and vice versa. After some trails, I am using faster_rcnn_inception_v2_coco for my project.

After deciding the model to be used, you will need an object detection training pipeline. They also provide sample config files on the repo. For my training, I will download faster_rcnn_inception_v2_coco.config.

Then you will need the dataset's (TFRecord files) corresponding label map. Example of how to create label maps can be found [here][9]. Here is also my label map which was very simple since I had only two classes, make a new file pascal_label_map.pbtxt which looks like this:
```
item {
  id: 1
  name: 'pos'
}

item {
  id: 2
  name: 'neg'
}
```

It is important to configure the faster_rcnn_inception_v2_coco.config file. You need to change it based on your configurations.

Change the number of classes in the file according to our requirement.
```
#before
num_classes: 90
#After
num_classes: 2
```

Change the total number of steps, depends on the complexity.
```
#before
num_steps: 200000
#After
num_steps: 1000
```

If your PC does not have good GPU then you need to decrease the batch_size.
```
#before
batch_size: 24
#After
batch_size: 1
```

Give the path to downloaded model i.e faster_rcnn_inception_v2_coco, the model we decided to be used.
```
#before
fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"
#after
fine_tune_checkpoint: "faster_rcnn_inception_v2_coco/model.ckpt"
```

Give the path to train.record file.
```
#before
train_input_reader: {  
tf_record_input_reader {   
input_path: "PATH_TO_BE_CONFIGURED/mscoco_train.record"
}
label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
}
#after
train_input_reader: {  
tf_record_input_reader {   
input_path: "data/train.record"
}
label_map_path: "data/pascal_label_map.pbtxt"
}
```

Give path for test.record file
```
#before
eval_input_reader: {  
tf_record_input_reader {
input_path: "PATH_TO_BE_CONFIGURED/mscoco_val.record" 
 }
label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"  shuffle: false
num_readers: 1}
#after
eval_input_reader: {  
tf_record_input_reader {
input_path: "data/test.record" 
 }
label_map_path: "data/pascal_label_map.pbtxt"  
shuffle: false
num_readers: 1}
```

Now, copy [train.py][10] from models/research/object-detection directory of the TensorFlow object detection repo.
```
python train.py --logtostderr --train_dir=train/ --pipeline_config_path=faster_rcnn_inception_v2_coco.config
```
If everything goes right, you will see the loss at a particular step.

Training can be either done locally or on the cloud (AWS, Google Cloud etc.). If you have a good GPU at home then you can do it locally otherwise I would recommend going with the cloud. In my case, a 3.8G Hz i5 processor takes about 2 hours for the training, still acceptable.


## Evaluating the model

The final step is to evaluate the trained model saved in train/ directory. You need to edit the faster_rcnn_inception_v2_coco.config file change to num_examples to the number of the Test Cases that be printed out of [xml_to_csv.py][11].
```
eval_config: {
  num_examples: 31
  # Note: The below line limits the evaluation process to 10 evaluations.
  # Remove the below line to evaluate indefinitely.
  max_evals: 10
}
```

You need to copy the [eval.py][12] file from the repo and evaluate using the following command:
```
python eval.py --logtostderr --pipeline_config_path=data/faster_rcnn_inception_v2_coco.config --checkpoint_dir=train/ --eval_dir=eval/
```
This will save the eval results in eval/ directory. To visualize the results we will use tensorboard.
To visualize the eval results
```
tensorboard --logdir=eval/
```
To visualize the training results
```
tensorboard --logdir=training/
```
Open the link in a browser and under Images tag you can see the results.


## Exporting the model

Copy the [exporter.py][13] and [export_inference_graph.py][14] from the object detection repo and run the following command, the number of steps depends on your configuration:
```
python export_inference_graph.py --pipeline_config_path=faster_rcnn_inception_v2_coco.config --output_directory=output --trained_checkpoint_prefix=train/model.ckpt-[NUMBER_OF_STEPS]
```

## Deployment of the model

You need to copy the utils folder from the object detection repo to the App folder and create a [app.py][15] file.
```python
#app.py

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2
import time

from object_detection.utils import ops as utils_ops

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
	raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

DIR = "[The Root Folder Path]"

from utils import label_map_util

from utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = os.path.join(DIR, 'output', 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(DIR, 'data/pascal_label_map.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')
		sess = tf.Session()
		# Get handles to input and output tensors
		ops = tf.get_default_graph().get_operations()
		all_tensor_names = {
			output.name for op in ops for output in op.outputs}
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
			# The following processing is only for single image
			detection_boxes = tf.squeeze(
				tensor_dict['detection_boxes'], [0])
			detection_masks = tf.squeeze(
				tensor_dict['detection_masks'], [0])
			# Reframe is required to translate mask from box coordinates to
			# image coordinates and fit the image size.
			real_num_detection = tf.cast(
				tensor_dict['num_detections'][0], tf.int32)
			detection_boxes = tf.slice(detection_boxes, [0, 0], [
									   real_num_detection, -1])
			detection_masks = tf.slice(detection_masks, [0, 0, 0], [
									   real_num_detection, -1, -1])
			detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
				detection_masks, detection_boxes, image.shape[0], image.shape[1])
			detection_masks_reframed = tf.cast(
				tf.greater(detection_masks_reframed, 0.5), tf.uint8)
			# Follow the convention by adding back the batch dimension
			tensor_dict['detection_masks'] = tf.expand_dims(
				detection_masks_reframed, 0)
		image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def run_inference_for_single_image(image):
	# Run inference
	output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

	# all outputs are float32 numpy arrays, so convert types as
	# appropriate
	output_dict['num_detections'] = int(
		output_dict['num_detections'][0])
	output_dict['detection_classes'] = output_dict[
		'detection_classes'][0].astype(np.uint8)
	output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
	output_dict['detection_scores'] = output_dict[
		'detection_scores'][0]
	if 'detection_masks' in output_dict:
		output_dict['detection_masks'] = output_dict[
			'detection_masks'][0]
	return output_dict

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
time.sleep(2)

while(True):
	start = time.time()
	ret, frame = cap.read()
	output_dict = run_inference_for_single_image(frame)
	vis_util.visualize_boxes_and_labels_on_image_array(
		frame,
		output_dict['detection_boxes'],
		output_dict['detection_classes'],
		output_dict['detection_scores'],
		category_index,
		instance_masks=output_dict.get('detection_masks'),
		use_normalized_coordinates=True,
		line_thickness=8)

	cv2.imshow('frame', frame)
	end = time.time()
	print('Time Taken: %f' % (end - start))

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
```

  [1]: https://www.tensorflow.org/install/
  [2]: https://github.com/tensorflow/models
  [3]: /img/bVbn0IO
  [4]: https://github.com/tzutalin/labelImg
  [5]: https://github.com/HiroshiFuu/Safety_Goggles_TensorFlow/blob/master/generate_tfrecord.py
  [6]: https://github.com/HiroshiFuu/Safety_Goggles_TensorFlow/blob/master/xml_to_csv.py
  [7]: https://github.com/HiroshiFuu/Safety_Goggles_TensorFlow/blob/master/xml_to_csv.py
  [8]: https://github.com/HiroshiFuu/Safety_Goggles_TensorFlow/blob/master/generate_tfrecord.py
  [9]: https://github.com/tensorflow/models/tree/master/research/object_detection/data
  [10]: https://github.com/HiroshiFuu/Safety_Goggles_TensorFlow/blob/master/train.py
  [11]: https://github.com/HiroshiFuu/Safety_Goggles_TensorFlow/blob/master/xml_to_csv.py
  [12]: https://github.com/HiroshiFuu/Safety_Goggles_TensorFlow/blob/master/eval.py
  [13]: https://github.com/HiroshiFuu/Safety_Goggles_TensorFlow/blob/master/exporter.py
  [14]: https://github.com/HiroshiFuu/Safety_Goggles_TensorFlow/blob/master/export_inference_graph.py
  [15]: https://github.com/HiroshiFuu/Safety_Goggles_TensorFlow/blob/master/App/demo.py
  [16]: https://github.com/HiroshiFuu/Safety_Goggles_TensorFlow
