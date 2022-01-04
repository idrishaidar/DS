import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def get_images_and_annotations(images_path, annot_path):
    # traverse through image and annotation files
    labels = []
    img_data = np.array([])
    img_names = []
    img_paths = []
    bndboxes = []

    # for each annotation file
    for annot_filename in os.listdir(annot_path):
        # get xml and png file path
        img_name = annot_filename.replace('.xml', '')
        xml_file_path = annot_path + annot_filename
        png_file_path = (images_path + annot_filename).replace('xml', 'png')
        
        # load the image to array
        img = tf.keras.preprocessing.image.load_img(png_file_path)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        # parse the xml file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # check if image contains more than one person
        object_elements = root.findall('object')
        if len(object_elements) == 1:
            # get image size to help rescaling later
            size_element = root.find('size')
            curr_width = int(size_element.find('width').text)
            curr_height = int(size_element.find('height').text)

            # get label
            object_element = object_elements[0]
            curr_label = object_element.find('name').text
            
            # get bounding box and rescale to 0 - 1
            bndbox_element = object_element.find('bndbox')
            xmin = int(bndbox_element.find('xmin').text) / curr_width
            ymin = int(bndbox_element.find('ymin').text) / curr_height
            xmax = int(bndbox_element.find('xmax').text) / curr_width
            ymax = int(bndbox_element.find('ymax').text) / curr_height
            curr_bndbox = np.array([[xmin, ymin, xmax, ymax]], dtype=np.float32)

            # append current image array, filename, path, label, and bounding box
            labels.append(curr_label)
            resized_img_array = tf.image.resize([img_array], [224,224]).numpy()[0]
            
            if len(img_data) == 0: 
                img_data = np.expand_dims(resized_img_array, axis=0)
            else:
                img_data = np.append(img_data, np.expand_dims(resized_img_array, axis=0), axis=0)
                
            img_names.append(img_name)
            img_paths.append(png_file_path)
            bndboxes.append(curr_bndbox)

    dataset = {
        'labels': labels,
        'img_data': img_data,
        'img_names': img_names,
        'img_paths': img_paths,
        'bndboxes': bndboxes
    }

    return dataset

def create_dataset_tensors(dataset):
    labels = dataset['labels']
    img_data = dataset['img_data']
    bndboxes = dataset['bndboxes']
    img_paths = dataset['img_paths']

    # label encoding
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # train test split
    splits = train_test_split(
        img_data, encoded_labels, bndboxes, img_paths,
        test_size=0.2, random_state=22, stratify=encoded_labels
    )

    train_images, test_images = splits[:2]
    train_labels, test_labels = splits[2:4]
    train_boxes, test_boxes = splits[4:6]
    train_paths, test_paths = splits[6:]

    # pass the image arrays to tf dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_images, (train_labels, train_boxes, train_paths))
    )
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_images, (test_labels, test_boxes, test_paths))
    )

    return train_dataset, test_dataset