#<-------------------------------------> Import <------------------------------------->
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np




#<---------------------------------> Internal Import <--------------------------------->
from DIPlib.enhancements import lpcSharpening, unsharpMasking
from DIPlib.features.regions import regionBasedFeatures
from DIPlib.filters.frequency.highpassFilter import highpassFilter
from DIPlib.intensityTransform import *
from DIPlib.fourier import *
from skimage.exposure import equalize_hist
from DIPlib.morphology import *
import skimage.morphology as skmorph
from glob import glob






#<-----------------------------------> Main Script <----------------------------------->
DATABASE_PATH = "input/Leaves/"

if __name__ == "__main__":
    #- Read image
    # input_img = cv.imread("input/fish_02.jpg")
    # rgb_img = cv.cvtColor(input_img, cv.COLOR_BGR2RGB)
    input_file_1 = glob(DATABASE_PATH + "1/" + "*")
    input_file_2 = glob(DATABASE_PATH + "2/" + "*")
    input_file = input_file_1 + input_file_2
    print(input_file)
    
    for f in input_file:
        input_img = cv.imread(f)
        rgb_img = cv.cvtColor(input_img, cv.COLOR_BGR2RGB)

        #- CRS
        gb_diff = rgb_img[:,:,1].astype(float) - rgb_img[:,:,2].astype(float)
        gb_diff = np.clip(gb_diff,0,255).astype(np.uint8)
        _, seg_img = cv.threshold(gb_diff,None,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        stre = skmorph.disk(9)

        morph_img = removeFragments(seg_img,thresh_ratio=0.05)
        morph_img = fillHoles(morph_img)
        morph_img = cv.morphologyEx(morph_img,cv.MORPH_CLOSE,stre)
        morph_img = fillHoles(morph_img)

        _,eccen = regionBasedFeatures(morph_img,"eccentricity")

        if eccen[0] < 0.8:
            leaf_class = "1"
        else:
            leaf_class = "2"

        plt.subplot(1,2,1)
        plt.imshow(rgb_img)
        plt.subplot(1,2,2)
        plt.title(f"Leaf Class: {leaf_class}")
        plt.imshow(morph_img, cmap="gray")
        plt.show()


    