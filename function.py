from imutils import face_utils
import dlib
import cv2
import imutils
import os
import re
import sys
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import argparse


def readimage(p,detector,predictor,width,color,msshow,
              face,mschoose,ir,keyd,keyl):

    image = cv2.imread(ir)
    image = imutils.resize(image, width)

    r = input(mschoose)

    gray = cv2.cvtColor(image, color)
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        if r == keyd:
            cv2.imshow(msshow, image)

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	    # show the face number
        cv2.putText(image, face .format(i + 1), (x - 10, y - 10),
	    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            if r == keyl:
               cv2.imshow(msshow, image)

        # show the output image with the face detections + facial landmarks
        
    cv2.waitKey(0)

def showlandmark(p,detector,predictor,width,color,keyq,
                 keyl,extension,msshow,id,ir,iwl):
    
    image = cv2.imread(ir)
    image = imutils.resize(image, width)
    gray = cv2.cvtColor(image, color)

    rects = detector(gray, 1)
    img_id = id

    while True:

        img_id += 1
        pressedKey = cv2.waitKey(1) & 0xFF
        for (i, rect) in enumerate(rects):

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
    
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            # show the output image with the face detections + facial landmarks
        if pressedKey == ord(keyl):
            cv2.imwrite(iwl +str(img_id)+extension, image)
        elif pressedKey == ord(keyq):
            break

        cv2.imshow(msshow, image)

    cv2.destroyAllWindows()
    cv2.waitKey(0)

def detectface(p,detector,predictor,color,keyq,keyl,extension,
               face,msshow,keyd,read,id,iwd,iwl):

    cap = read
    img_id = id
    while True:
        
        # load the input image and convert it to grayscale
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, color)
        
        # detect faces in the grayscale image
        rects = detector(gray, 0)
        pressedKey = cv2.waitKey(1) & 0xFF
        # loop over the face detections

        for (i, rect) in enumerate(rects):
            
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            if pressedKey == ord(keyd):
                cv2.imwrite(iwd +str(img_id)+extension, frame)
                img_id += 1

            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	        # show the face number
            cv2.putText(frame, face.format(i + 1), (x - 10, y - 10),
		    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            # show the output image with the face detections + facial landmarks
        if pressedKey == ord(keyl):
            cv2.imwrite(iwl +str(img_id )+extension, frame)
            img_id += 1
        elif pressedKey == ord(keyq):
            break

        cv2.imshow(msshow, frame)
        
    
    cv2.destroyAllWindows()
    cap.release()
    
def checkerror(mserror,repart,renum,dir,newmodel,dlibmodel,filexml):

    REG_PART = re.compile(repart)
    REG_NUM = re.compile(renum)

    # dataset path
    ibug_dir = dir
 
    def measure_model_error(model, xml_annotations):
        '''requires: the model and xml path.
        It measures the error of the model on the given
        xml file of annotations.'''
        error = dlib.test_shape_predictor(xml_annotations, model)
        print(mserror.format(model, error))


    measure_model_error(newmodel, filexml)
    measure_model_error(dlibmodel, filexml)


    def test(image_path, model_path):
        '''Test the given model by showing the detected landmarks.
            - image_path: the path of an image. Should contain a face.
            - model_path: the path of a shape predictor model.
        '''
        image = cv2.imread(image_path)
        face_detector = dlib.get_frontal_face_detector()
        dets = face_detector(image, 1)
        predictor = dlib.shape_predictor(model_path)

        for d in dets:
            cv2.rectangle(image, (d.left(), d.top()), (d.right(), d.bottom()), 255, 1)
            shape = predictor(image, d)
            for i in range(shape.num_parts):
                p = shape.part(i)
                cv2.circle(image, (p.x, p.y), 2, 255, 1)
                cv2.putText(image, str(i), (p.x + 4, p.y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255))

        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def replacevalue(dir,rpwhat,rpwith,readpath):

    my_dir = dir
    replace_what = rpwhat
    replace_with = rpwith

    # loop through all files in directory
    for fn in os.listdir(my_dir):
    #print(fn)
        pathfn = os.path.join(my_dir,fn)
        if os.path.isfile(pathfn):
            file = open(pathfn, readpath)
            new_file_content=''
        for line in file:
            p = re.compile(replace_what)
            new_file_content += p.sub(replace_with, line)
        file.seek(0)
        file.truncate()
        file.write(new_file_content)
        file.close()

def trainmodel(mserror,repart,renum,eye,nose,face,landmark,train,test,opt,tree,nu,
               cascade,feature,ntest,osamount,ostranslation,be,nthreads,namemodel):

    # regex:
    REG_PART = re.compile(repart)
    REG_NUM = re.compile(renum)

    # landmarks subsets (relative to 68-landmarks):
    EYE_EYEBROWS = eye
    NOSE_MOUTH = nose
    FACE_CONTOUR = face
    ALL_LANDMARKS = landmark


    # annotations
    train_labels = train
    test_labels = test

    def slice_xml(in_path, out_path, parts):
        '''creates a new xml file stored at [out_path] with the desired landmark-points.
        The input xml [in_path] must be structured like the ibug annotation xml.'''
        file = open(in_path, "r")
        out = open(out_path, "w")
        pointSet = set(parts)

        for line in file.readlines():
            finds = re.findall(REG_PART, line)

            # find the part section
            if len(finds) <= 0:
                out.write(line)
            else:
                # we are inside the part section 
                # so we can find the part name and the landmark x, y coordinates
                name, x, y = re.findall(REG_NUM, line)

                # if is one of the point i'm looking for, write in the output file
                if int(name) in pointSet:
                    out.write(f"      <part name='{name}' x='{x}' y='{y}'/>\n")

        out.close()

    def train_model(name, xml):
        '''requires: the model name, and the path to the xml annotations.
        It trains and saves a new model according to the specified 
        training options and given annotations'''
        # get the training options
        options = opt
        options.tree_depth = tree
        options.nu = nu
        options.cascade_depth = cascade
        options.feature_pool_size = feature
        options.num_test_splits = ntest
        options.oversampling_amount = osamount
        options.oversampling_translation_jitter = ostranslation

        options.be_verbose = be  # tells what is happening during the training
        options.num_threads = nthreads   # number of the threads used to train the model
  
        # finally, train the model
        dlib.train_shape_predictor(xml, name, options)

    def measure_model_error(model, xml_annotations):
        '''requires: the model and xml path.
        It measures the error of the model on the given
        xml file of annotations.'''
        error = dlib.test_shape_predictor(xml_annotations, model)
        print(mserror.format(model, error))

    # -----------------------------------------------------------------------------
    # -- Model Generation
    # -----------------------------------------------------------------------------

    # add or remove models here.
    models = [
     # pair: model name, parts
    (namemodel, ALL_LANDMARKS)
    ]

    for model_name, parts in models:
        print(f"processing model: {model_name}")
  
        train_xml = f"{model_name}_train.xml"
        test_xml = f"{model_name}_test.xml"
        dat = f"{model_name}.dat"
        slice_xml(train_labels, train_xml, parts)
        slice_xml(test_labels, test_xml, parts)
  
        print("train")
        # training
        train_model(dat, train_xml)
 
        # compute traning and test error
        measure_model_error(dat, train_xml)
        measure_model_error(dat, test_xml)

        # -----------------------------------------------------------------------------

    def test(image_path, model_path):
        '''Test the given model by showing the detected landmarks.
            - image_path: the path of an image. Should contain a face.
            - model_path: the path of a shape predictor model.
        '''
        image = cv2.imread(image_path)
        face_detector = dlib.get_frontal_face_detector()
        dets = face_detector(image, 1)
        predictor = dlib.shape_predictor(model_path)

        for d in dets:
            cv2.rectangle(image, (d.left(), d.top()), (d.right(), d.bottom()), 255, 1)
            shape = predictor(image, d)
            for i in range(shape.num_parts):
                p = shape.part(i)
                cv2.circle(image, (p.x, p.y), 2, 255, 1)
                cv2.putText(image, str(i), (p.x + 4, p.y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255))

        cv2.imshow("window", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def imagejitter(default,msload,rotation,zoom,width,height,shear,status,mode,
                total,msgenerat,file):

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
	    help="path to the input image")
    ap.add_argument("-o", "--output", required=True,
	    help="path to output directory to store augmentation examples")
    ap.add_argument("-t", "--total", type=int, default=default,
	    help="# of training samples to generate")
    args = vars(ap.parse_args())

    # load the input image, convert it to a NumPy array, and then
    # reshape it to have an extra dimension
    print(msload)
    image = load_img(args["image"])
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # construct the image generator for data augmentation then
    # initialize the total number of images generated thus far
    aug = ImageDataGenerator(
	    rotation_range = rotation,
	    zoom_range = zoom,
	    width_shift_range = width,
	    height_shift_range = height,
	    shear_range = shear,
	    horizontal_flip = status,
	    fill_mode = mode)
    total = total

    # construct the actual Python generator
    print(msgenerat)
    imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"],
	    save_prefix="image", save_format=file)

    # loop over examples from our image data augmentation generator
    for image in imageGen:
	    # increment our counter
	    total += 1

	    # if we have reached the specified number of examples, break
	    # from the loop
	    if total == args["total"]:
		    break

