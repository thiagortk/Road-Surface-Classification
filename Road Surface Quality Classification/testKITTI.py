import cv2 as cv
import numpy as np
import tensorflow as tf
import argparse
import sys
import os.path
import random
import os
import glob
import operator
import time

image_size=128
num_channels=3
images = []

outputFile = sys.argv[2]

# Opening frames
cap = cv.VideoCapture(sys.argv[1])

vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 15, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

width = round(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

newHeight = int(round(height/2))

graph = tf.Graph()
graphAQ = tf.Graph()
graphPQ = tf.Graph()
graphUQ = tf.Graph()

default_graph = tf.get_default_graph()

# ----------------------------- #
# Restoring the model for types #
# ----------------------------- #
with graph.as_default():
    saver = tf.train.import_meta_graph('roadsurfaceType-model.meta')
    # Acessing the graph
    #
    y_pred = graph.get_tensor_by_name("y_pred:0")

    #
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, len(os.listdir('training_data_type'))))

sess = tf.Session(graph = graph)
saver.restore(sess, tf.train.latest_checkpoint('typeCheckpoint/'))


# --------------------------------------- #
# Restoring the model for asphalt quality #
# --------------------------------------- #
with graphAQ.as_default():
    saverAQ = tf.train.import_meta_graph('roadsurfaceAsphaltQuality-model.meta')
    # Acessing the graph
    #
    y_predAQ = graphAQ.get_tensor_by_name("y_pred:0")

    #
    xAQ = graphAQ.get_tensor_by_name("x:0")
    y_trueAQ = graphAQ.get_tensor_by_name("y_true:0")
    y_test_imagesAQ = np.zeros((1, len(os.listdir('training_data_asphalt_quality'))))

sessAQ = tf.Session(graph = graphAQ)
saverAQ.restore(sessAQ, tf.train.latest_checkpoint('asphaltCheckpoint/'))


# ------------------------------------- #
# Restoring the model for paved quality #
# ------------------------------------- #
with graphPQ.as_default():
    saverPQ = tf.train.import_meta_graph('roadsurfacePavedQuality-model.meta')
    # Acessing the graph
    #
    y_predPQ = graphPQ.get_tensor_by_name("y_pred:0")

    #
    xPQ = graphPQ.get_tensor_by_name("x:0")
    y_truePQ = graphPQ.get_tensor_by_name("y_true:0")
    y_test_imagesPQ = np.zeros((1, len(os.listdir('training_data_paved_quality'))))

sessPQ = tf.Session(graph = graphPQ)
saverPQ.restore(sessPQ, tf.train.latest_checkpoint('pavedCheckpoint/'))


# --------------------------------------- #
# Restoring the model for unpaved quality #
# --------------------------------------- #
with graphUQ.as_default():
    saverUQ = tf.train.import_meta_graph('roadsurfaceUnpavedQuality-model.meta')
    # Acessing the graph
    #
    y_predUQ = graphUQ.get_tensor_by_name("y_pred:0")

    #
    xUQ = graphUQ.get_tensor_by_name("x:0")
    y_trueUQ = graphUQ.get_tensor_by_name("y_true:0")
    y_test_imagesUQ = np.zeros((1, len(os.listdir('training_data_unpaved_quality'))))

sessUQ = tf.Session(graph = graphUQ)
saverUQ.restore(sessUQ, tf.train.latest_checkpoint('unpavedCheckpoint/'))


while cv.waitKey(1) < 0:

    hasFrame, images = cap.read()

    finalimg = images

    if not hasFrame:
        print("Classification done!")
        print("Results saved as: ", outputFile)
        cv.waitKey(3000)
        break

    images = images[newHeight-5:height-50, 0:width]
    images = cv.resize(images, (image_size, image_size), 0, 0, cv.INTER_LINEAR)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0)

    x_batch = images.reshape(1, image_size, image_size, num_channels)

    #
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)


    outputs = [result[0,0], result[0,1], result[0,2]]

    value = max(outputs)
    index = np.argmax(outputs)


    if index == 0: #Asphalt
        label = 'Asphalt'
        prob = str("{0:.2f}".format(value))
        color = (0, 0, 0)
        x_batchAQ = images.reshape(1, image_size, image_size, num_channels)
        #
        feed_dict_testingAQ = {xAQ: x_batchAQ, y_trueAQ: y_test_imagesAQ}
        resultAQ = sessAQ.run(y_predAQ, feed_dict=feed_dict_testingAQ)
        outputsQ = [resultAQ[0,0], resultAQ[0,1], resultAQ[0,2]]
        valueQ = max(outputsQ)
        indexQ = np.argmax(outputsQ)
        if indexQ == 0: #Asphalt - Good
            quality = 'Good'
            colorQ = (0, 255, 0)
            probQ =  str("{0:.2f}".format(valueQ))
        elif indexQ == 1: #Asphalt - Regular
            quality = 'Regular'
            colorQ = (0, 204, 255)
            probQ =  str("{0:.2f}".format(valueQ))
        elif indexQ == 2: #Asphalt - Bad
            quality = 'Bad'
            colorQ = (0, 0, 255)
            probQ =  str("{0:.2f}".format(valueQ))  
    elif index == 1: #Paved
        label = 'Paved'
        prob = str("{0:.2f}".format(value))
        color = (153, 102, 102)
        x_batchPQ = images.reshape(1, image_size, image_size, num_channels)
        #
        feed_dict_testingPQ = {xPQ: x_batchPQ, y_truePQ: y_test_imagesPQ}
        resultPQ = sessPQ.run(y_predPQ, feed_dict=feed_dict_testingPQ)
        outputsQ = [resultPQ[0,0], resultPQ[0,1], resultPQ[0,2]]
        valueQ = max(outputsQ)
        indexQ = np.argmax(outputsQ)
        if indexQ == 0: #Paved - Good
            quality = 'Good'
            colorQ = (0, 255, 0)
            probQ =  str("{0:.2f}".format(valueQ))
        elif indexQ == 1: #Paved - Regular
            quality = 'Regular'
            colorQ = (0, 204, 255)
            probQ =  str("{0:.2f}".format(valueQ))
        elif indexQ == 2: #Paved - Bad
            quality = 'Bad'
            colorQ = (0, 0, 255)
            probQ =  str("{0:.2f}".format(valueQ))
    elif index == 2: #Unpaved
        label = 'Unpaved'
        prob = str("{0:.2f}".format(value))
        color = (0, 153, 255)
        x_batchUQ = images.reshape(1, image_size, image_size, num_channels)
        #
        feed_dict_testingUQ = {xUQ: x_batchUQ, y_trueUQ: y_test_imagesUQ}
        resultUQ = sessUQ.run(y_predUQ, feed_dict=feed_dict_testingUQ)
        outputsQ = [resultUQ[0,0], resultUQ[0,1]]
        valueQ = max(outputsQ)
        indexQ = np.argmax(outputsQ)
        if indexQ == 0: #Unpaved - Regular
            quality = 'Regular'
            colorQ = (0, 204, 255)
            probQ =  str("{0:.2f}".format(valueQ))
        elif indexQ == 1: #Unpaved - Bad
            quality = 'Bad'
            colorQ = (0, 0, 255)
            probQ =  str("{0:.2f}".format(valueQ))

    cv.rectangle(finalimg, (0*2, 0*2), (145*2, 80*2), (255, 255, 255), cv.FILLED)
    cv.putText(finalimg, 'Class: ', (5*2,15*2), cv.FONT_HERSHEY_SIMPLEX, 0.5*2, (0,0,0), 2)
    cv.putText(finalimg, label, (70*2,15*2), cv.FONT_HERSHEY_SIMPLEX, 0.5*2, color, 2)
    cv.putText(finalimg, prob, (5*2,35*2), cv.FONT_HERSHEY_SIMPLEX, 0.5*2, (0,0,0), 2)
    cv.putText(finalimg, 'Quality: ', (5*2,55*2), cv.FONT_HERSHEY_SIMPLEX, 0.5*2, (0,0,0), 2)
    cv.putText(finalimg, quality, (70*2,55*2), cv.FONT_HERSHEY_SIMPLEX, 0.5*2, colorQ, 2)
    cv.putText(finalimg, probQ, (5*2,75*2), cv.FONT_HERSHEY_SIMPLEX, 0.5*2, (0,0,0), 2)

    vid_writer.write(finalimg.astype(np.uint8))

sess.close()
sessAQ.close()
sessPQ.close()
sessUQ.close()
time.sleep(5)
