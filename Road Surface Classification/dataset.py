import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


def adjust_gamma(image):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
        gamma = 0.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
 
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Going to read training images')
    for fields in classes:   
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)

            # Region Of Interest (ROI)
            height, width = image.shape[:2]
            newHeight = int(round(height/2))
            image = image[newHeight-5:height-50, 0:width]

            brght_img = increase_brightness(image, value=150)

            shaded_img = adjust_gamma(image)
            
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)

            brght_img = cv2.resize(brght_img, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            brght_img = brght_img.astype(np.float32)
            brght_img = np.multiply(brght_img, 1.0 / 255.0)

            shaded_img = cv2.resize(shaded_img, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            shaded_img = shaded_img.astype(np.float32)
            shaded_img = np.multiply(brght_img, 1.0 / 255.0)

            if index == 0:
                images.append(image)
                images.append(brght_img)
                images.append(shaded_img)

                label = np.zeros(len(classes))
                label[index] = 1.0

                labels.append(label)       
                labels.append(label)
                labels.append(label)

                flbase = os.path.basename(fl)

                img_names.append(flbase)
                img_names.append(flbase)
                img_names.append(flbase)

                cls.append(fields)
                cls.append(fields)
                cls.append(fields)
            elif index == 1:
                for i in range(3):
                    images.append(image)
                    images.append(brght_img)
                    images.append(shaded_img)

                    label = np.zeros(len(classes))
                    label[index] = 1.0

                    labels.append(label)       
                    labels.append(label)
                    labels.append(label)

                    flbase = os.path.basename(fl)

                    img_names.append(flbase)
                    img_names.append(flbase)
                    img_names.append(flbase)

                    cls.append(fields)
                    cls.append(fields)
                    cls.append(fields)
            elif index == 2:
                for i in range(6):
                    images.append(image)
                    images.append(brght_img)
                    images.append(shaded_img)

                    label = np.zeros(len(classes))
                    label[index] = 1.0

                    labels.append(label)       
                    labels.append(label)
                    labels.append(label)

                    flbase = os.path.basename(fl)

                    img_names.append(flbase)
                    img_names.append(flbase)
                    img_names.append(flbase)

                    cls.append(fields)
                    cls.append(fields)
                    cls.append(fields)
            
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls


class DataSet(object):

  def __init__(self, images, labels, img_names, cls):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, img_names, cls = load_train(train_path, image_size, classes)
  images, labels, img_names, cls = shuffle(images, labels, img_names, cls)  

  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_img_names = img_names[:validation_size]
  validation_cls = cls[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_img_names = img_names[validation_size:]
  train_cls = cls[validation_size:]

  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

  return data_sets
