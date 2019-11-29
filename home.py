import function
import cv2
import dlib

#detection
p = "face_landmarks_18-11-62.dat";
detector = dlib.get_frontal_face_detector();
predictor = dlib.shape_predictor(p);
#resize
width = 500;
#color
color = cv2.COLOR_BGR2GRAY;
#key exit
keyq = ('q')
#key save image landmark
keyl = ('l')
#print error model
mserror = "error of the model: {} is {}"
#message imshow
msshow = "output"
#message count face
face = "face #{}"
#chang File extension
extension = ".jpg"


#functuion readimage
"""function.readimage (
    p,
    detector,
    predictor,
    width,
    color,
    msshow,
    face,
    mschoose = "please choose d|l: ",   #print choose
    ir = "test/te1.jpg",                #imread image
    keyd = ('d'),                       #imshow image detection
    keyl = ('l')                        #imshow image landmark
    
    );"""

#function showlandmark
"""function.showlandmark (
    p,
    detector,
    predictor,
    width, 
    color,
    keyq,
    keyl,
    extension,
    msshow,
    id = 0,                             #img_id
    ir = "picture/train/t5.jpg",        #imread image 
    iwl = "picture/savelandmark/la."    #imwrite image landmark

    );"""

#function detectface
"""function.detectface (
    p,
    detector,
    predictor,
    color,
    keyq,
    keyl,
    extension,
    face,
    msshow,
    keyd = ('d'),                    #key save image detection
    read = cv2.VideoCapture(0),
    id = 100,                        #img_id
    iwd = "picture/detection/de.",   #imwrite image detection
    iwl = "picture/landmark/la."     #imwrite image landmark

    );"""

#function checkerror
"""function.checkerror(
    mserror,                                          #print error model
    repart = "part name='[0-9]+'",                    #REG_PART
    renum = "[0-9]+",                                 #REG_NUM
    dir = "face_landmark",                            #ibug_dir
    newmodel = "face_landmarks_18-11-62.dat",         #file .dat new model
    dlibmodel = "face_landmarks_68_30.dat",           #file .dat dlib model
    filexml = f"face_landmarks_18-11-62_test.xml"     #file .xml test&train model

    );"""

#function replacevalue
"""function.replacevalue(
    dir = 'C://Users//FRONTIS//OneDrive - Frontis//face_landmark//filereplace',   #my_dir
    rpwhat = "name='0'",     #replacewhat
    rpwith = "name='00'",    #replacewith
    readpath = 'r+'          #file open

    );"""

#function trainmodel
"""function.trainmodel(
    mserror,                                                #print error model
    repart = "part name='[0-9]+'",                          #REG_PART
    renum = "[0-9]+",                                       #REG_NUM
    eye = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 36,      #EYE_EYEBROWS
           37, 38, 39,40, 41, 42, 43, 44, 45, 46, 47],
    nose = [27, 28, 29, 30, 31, 32, 33, 34, 35, 48, 49,     #NOSE_MOUTH
            50, 51, 52,53, 54, 55, 56, 57, 58, 59, 60, 
            61, 62, 63, 64, 65, 66, 67],
    face = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,   #FACE_CONTOUR
            14, 15, 16],  
    landmark = range(0, 68),                                #ALL_LANDMARKS           
    train = "labels_ibug_300W_train.xml",                   #file .xml train model
    test = "face_landmarks_test.xml",                       #file .xml test model
    opt = dlib.shape_predictor_training_options(),          #option
    tree = 3,                                               #options.tree_depth
    nu = 0.1,                                               #options.nu
    cascade = 10,                                           #options.cascade_depth
    feature = 150,                                          #options.feature_pool_size
    ntest = 350,                                            #options.num_test_splits
    osamount = 5,                                           #options.oversampling_amount
    ostranslation = 0,                                      #options.oversampling_translation_jitter
    be = True,                                              #options.be_verbose
    nthreads = 1,                                           #options.num_threads
    namemodel = "face_landmarks_",                      #model name
    
    );"""
