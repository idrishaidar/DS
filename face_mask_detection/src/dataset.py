import numpy as np
import xml.etree.ElementTree as ET
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow import image, data

class ImageDataset:
    def __init__(self, images_path: str, annot_path: str):
        self.images_path = images_path
        self.annot_path = annot_path
        self.labels = []
        self.img_data = np.array([])
        self.img_names = []
        self.img_paths = []
        self.bndboxes = []

        self.get_images_and_annot()

    def get_images_and_annot(self):
        # traverse through image and annotation files
        # for each annotation file
        for annot_filename in tqdm(os.listdir(self.annot_path)):
            # get xml and png file path
            img_name = annot_filename.replace('.xml', '')
            xml_file_path = self.annot_path + annot_filename
            png_file_path = (self.images_path + annot_filename).replace('xml', 'png')
            
            # load the image to array
            img = load_img(png_file_path)
            img_array = img_to_array(img)
            
            # parse the xml file
            root = ET.parse(xml_file_path).getroot()

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
                self.labels.append(curr_label)
                resized_img_array = image.resize([img_array], [224,224]).numpy()[0]
                
                if len(self.img_data) == 0: 
                    self.img_data = np.expand_dims(resized_img_array, axis=0)
                else:
                    self.img_data = np.append(self.img_data, np.expand_dims(resized_img_array, axis=0), axis=0)
                    
                self.img_names.append(img_name)
                self.img_paths.append(png_file_path)
                self.bndboxes.append(curr_bndbox)

    def create_dataset_tensors(self):
        # label encoding
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(self.labels)

        # train test split
        splits = train_test_split(
            self.img_data, encoded_labels, self.bndboxes, self.img_paths,
            test_size=0.2, random_state=22, stratify=encoded_labels
        )

        train_images, test_images = splits[:2]
        train_labels, test_labels = splits[2:4]
        train_boxes, test_boxes = splits[4:6]
        train_paths, test_paths = splits[6:]

        # pass the image arrays to tf dataset
        train_dataset = data.Dataset.from_tensor_slices(
            (train_images, (train_labels, train_boxes))
        )
        test_dataset = data.Dataset.from_tensor_slices(
            (test_images, (test_labels, test_boxes))
        )

        return train_dataset, test_dataset

# # dataset path
# images_path = '../dataset/images/'
# annot_path = '../dataset/annotations/'

# face_mask_dataset = ImageDataset(images_path, annot_path)
# train_dataset, test_dataset = face_mask_dataset.create_dataset_tensors()