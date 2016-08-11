# -*- coding: utf-8 -*-
from PIL import Image

import numpy as np
import pylab
import sys
import glob
import os
import pca


class EigenFaces(object):
    def read_images(self, path, sz=None):
        """Reads the images in a given folder, resizes images on the fly if size is given.

        Args:
            path: Path to a folder with subfolders representing the subjects (persons).
            sz: A tuple with the size Resizes

        Returns:
            A list [X,y]

                X: The images, which is a Python list of numpy arrays.
                y: The corresponding labels (the unique number of the subject, person).
        """
        classSamplesList = []
        class_matrices_list = []
        X,y = [], []
        for dirname, dirnames, filenames in os.walk(path):
            for subdirname in dirnames:
                subject_path = os.path.join(dirname, subdirname)
                classSamplesList = []
                for filename in os.listdir(subject_path):
                    if filename != ".DS_Store":
                        try:
                            im = Image.open(os.path.join(subject_path, filename))
                            # resize to given size (if given) e.g., sz = (480, 640)
                            if (sz is not None):
                                im = im.resize(sz, Image.ANTIALIAS)
                            X.append(np.asarray(im, dtype = np.uint8))

                        except IOError as e:
                            errno, strerror = e.args
                            print("I/O error({0}): {1}".format(errno, strerror))
                        except:
                            print("Unexpected error:", sys.exc_info()[0])
                            raise
                        # adds each sample within a class to this List
                        classSamplesList.append(np.asarray(im, dtype = np.uint8))

                # flattens each sample within a class and adds the array/vector to a class matrix
                class_samples_matrix = np.array([img.flatten()
                    for img in classSamplesList],'f')

                 # adds each class matrix to this MASTER List
                class_matrices_list.append(class_samples_matrix)

                y.append(subdirname)

        self.number_of_classes = len(class_matrices_list)

        # returns the images as a List of arrays; returns the class matrices to be projected and averaged
        return [X,y], class_matrices_list

    def train(self, root_training_images_folder):
        list_of_arrays_of_images = []
        self.labels_list = []
        list_of_matrices_of_flattened_class_samples = []

        ti = []
        self.projected_classes = []

        # read_images  returns X as a list of arrays of the Images AND y as a list of labels
        [list_of_arrays_of_images, self.labels_list], list_of_matrices_of_flattened_class_samples = self.read_images(root_training_images_folder)

        anImage = np.array(Image.fromarray(list_of_arrays_of_images[0]))
        m,n = anImage.shape[0:2] # get the size of the images

         # create matrix to store all flattened images
        images_matrix = np.array([np.array(Image.fromarray(im)).flatten()
              for im in list_of_arrays_of_images],'f')

        # perform PCA
        self.eigenfacesMatrix, variance, self.mean_Image = pca.pca(images_matrix)

        # Projecting each class sample (as class matrix) and then using the class average as the class weights for comparison with the Target image
        numberOfClasses = len(list_of_matrices_of_flattened_class_samples)

        for i in range(numberOfClasses):
            class_weights_vertex = self.projectImage(list_of_matrices_of_flattened_class_samples[i])
            self.projected_classes.append(class_weights_vertex.mean(0))

        # get a target image and flatten it
        target_images = self.getTargetImages()
        ti = np.array(Image.open(target_images[0]), dtype = np.uint8).flatten()

        print(self.predictFace(ti))

        #######################
        pylab.figure()

        pylab.gray()

        pylab.subplot(2,4,1)

        pylab.imshow(self.mean_Image.reshape(m,n))

        for i in range(7):
            pylab.subplot(2,4,i+2)
            pylab.imshow(self.eigenfacesMatrix[i].reshape(m,n))

    def extract(self,X):
        X = np.asarray(X).reshape(-1,1)
        return self.projectImage(X)

    def projectImage(self, X):
        X = X - self.mean_Image
        return np.dot(X, self.eigenfacesMatrix.T)

    def reconstruct(self, X):
        X = np.dot(X, self.eigenfacesMatrix)
        return X + self.mean_Image

    def getTargetImages(self):
        targetImageList = glob.glob('target_image/*.pgm')  # folder containing the traget image
        return targetImageList

    def predictFace(self, X):
        minClass = -1
        minDistance = np.finfo('float').max
        projected_target = self.projectImage(X)
        # delete last array item, it's nan
        projected_target = np.delete(projected_target, -1)
        for i in range(len(self.projected_classes)):
            distance = np.linalg.norm(projected_target - np.delete(self.projected_classes[i], -1))
            if distance < minDistance:
                minDistance = distance
                minClass = self.labels_list[i]
        predictedImg = "training_images/%s/1.pgm" % (minClass)
        img = Image.open(predictedImg)
        img.show()
        return minClass

    def predictRace(self, X):
        return np.minTarget

    def getClassAverageFromSamples(classSamples):
        m, n = np.array(classSamples).shape[1:3]
        l = len(classSamples)
        addSamplesTogether = np.zeros((m,n))

        for a in classSamples:
            addSamplesTogether = np.add(addSamplesTogether, a)

        averagedClass = np.divide(addSamplesTogether, l)

        return averagedClass

    def __repr__(self):
        return "PCA (num_components=%d)" % (self._num_components)
