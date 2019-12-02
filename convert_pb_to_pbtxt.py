import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format

filename = "./yolov3_coco.pb"

with gfile.FastGFile(filename,'rb') as f:
    graph_def = tf.GraphDef()
 
    graph_def.ParseFromString(f.read())
 
    tf.import_graph_def(graph_def, name='')
 
    tf.train.write_graph(graph_def, './', 'yolov3_coco.pbtxt', as_text=True)