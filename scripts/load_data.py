#This script loads and prepares the data set  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging 

import os 
from distutils.dir_util import copy_tree, remove_tree
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from random import randint



logging.basicConfig(filename='../logs/load_data.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.INFO)
data_directory = '../data'
new_directory = os.path.join(data_directory,'alldata')

class PREP:
    

    def __init__(self):
        pass


    def pre_process(self):

        try:

            test_directory = os.path.join(data_directory,'test')
            train_directory = os.path.join(data_directory,'train')
            print(" ================ Folder found ================= \n")
            print(" ==== Contents of train directory ========= \n",os.listdir(train_directory),"\n")
            print(" ==== Contents of test directory ========= \n",os.listdir(test_directory),"\n")

        except FileNotFoundError as e:
            logging.info(' !!! Error the file was not found !!!! ')
            print("!!!! Error the file path doesn't exist !!!! {} ".format(e.__class__))


        print("================ Creating new working tree ================")
        # os.mkdir(new_directory)
        # copy_tree(train_directory, new_directory)
        # copy_tree(test_directory, new_directory)

        #create a new folder and store all the data in it
        print("========= Creating directory ========= \n")
        print(" ============ new directory contains ======== \n")
        logging.info("========= created  directory =========")
        logging.info("os.listdir(new_directory)")
        # print(os.listdir(new_directory))


        #The images have been classified in the respective folders and we shall use thos as classes
        classes=os.listdir(new_directory)
        print(classes)

        IMG_SIZE=176
        IMAGE_SIZE=[176,176]
        DIM= (IMG_SIZE,IMG_SIZE)

        #we use image Data Generator from keras to load the images 

        ZOOM = [.99, 1.01]
        BRIGHT_RANGE = [0.8, 1.2]
        HORZ_FLIP = True
        FILL_MODE = "constant"
        DATA_FORMAT = "channels_last"

        work_dr = IDG(rescale = 1./255, brightness_range=BRIGHT_RANGE, zoom_range=ZOOM, data_format=DATA_FORMAT, fill_mode=FILL_MODE, horizontal_flip=HORZ_FLIP)

        train_data_gen = work_dr.flow_from_directory(directory=new_directory, target_size=DIM, batch_size=6500, shuffle=False)

        return train_data_gen,classes


