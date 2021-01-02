# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)
        

flags = tf.compat.v1.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'person':
        return 1
    if row_label == 'cup':
        return 2
    if row_label == 'food':
        return 3
    if row_label == 'tree':
        return 4
    if row_label == 'sky':
        return 5
    if row_label == 'pizza':
        return 6
    if row_label == 'knife':
        return 7
    if row_label == 'sweets':
        return 8
    if row_label == 'home':
        return 9
    if row_label == 'cheese':
        return 10
    if row_label == 'ring':
        return 11
    if row_label == 'stop':
        return 12
    if row_label == 'sandwitch':
        return 13
    if row_label == 'dolly':
        return 14
    if row_label == 'spool':
        return 15
    if row_label == 'watch':
        return 16
    if row_label == 'dog':
        return 17
    if row_label == 'cat':
        return 18
    if row_label == 'floor':
        return 19
    if row_label == 'tv':
        return 20
    if row_label == 'window':
        return 21
    if row_label == 'pc':
        return 22
    if row_label == 'shoes':
        return 23
    if row_label == 'ball':
        return 24
    if row_label == 'giraffe':
        return 25
    if row_label == 'chair':
        return 26
    if row_label == 'vase':
        return 27
    if row_label == 'hand':
        return 28
    if row_label == 'ring':
        return 29
    if row_label == 'pot':
        return 30
    if row_label == 'table':
        return 31
    if row_label == 'cofee':
        return 32
    if row_label == 'soupe':
        return 33
    if row_label == 'salade':
        return 34
    if row_label == 'cheese':
        return 35
    if row_label == 'bed':
        return 36
    if row_label == 'ciseaux':
        return 37
    if row_label == 'stop':
        return 38
    if row_label == 'home':
        return 39
    if row_label == 'books':
        return 40
    if row_label == 'sofa':
        return 39
    if row_label == 'mirror':
        return 39
    else:
        None



def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.compat.v1.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    del_all_flags( tf.compat.v1.flags.FLAGS)

    tf.compat.v1.app.run()