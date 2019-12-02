#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-01-20 16:06:06
#   Description :
#
#================================================================

import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image

return_elements = ["Placeholder:0", "concat_7:0", "mul_4:0"]
pb_file         = "./yolov3_tiny.pb"
image_path      = "./docs/images/noised-1.jpg"
num_classes     = 1
input_size      = 416
graph           = tf.Graph()

original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]
image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...]

return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

with tf.Session(graph=graph) as sess:
    pred_sbbox, pred_mbbox = sess.run(
        [return_tensors[1], return_tensors[2]],
                feed_dict={ return_tensors[0]: image_data})

print(pred_sbbox.shape)
print(pred_mbbox.shape)

pred_bbox = np.reshape(np.concatenate([pred_sbbox, pred_mbbox], axis=2), (-1, 4 + num_classes))

print(pred_bbox.shape)

bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
bboxes = utils.nms(bboxes, 0.45, method='nms')
image = utils.draw_bbox(original_image, bboxes)
image = Image.fromarray(image)
image.show()