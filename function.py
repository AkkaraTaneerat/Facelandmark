import cv2
import imutils
import dlib
import re
from imutils import face_utils


class myfunction:

    def __init__(

        self,
        detection,
        prediction,     
        read,
        count,
        keyd,
        keyl,
        keyq,
        write,
        newmodel,
        oldmodel,
        test,
        
        ):

        self.detection = detection
        self.prediction = prediction
        self.read = read
        self.count = count
        self.keyd = keyd
        self.keyl = keyl
        self.keyq = keyq
        self.write = write
        self.newmodel = newmodel
        self.oldmodel = oldmodel
        self.test = test

    def readimage(self):

        image = self.read
        receive = input("please select d or l: ")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detect = self.detection(gray, 1)

        for (i, rectangle) in enumerate(detect):

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.prediction(gray, rectangle)
            shape = face_utils.shape_to_np(shape)

            #show the output image with the face detections
            if receive == self.keyd:
                cv2.imshow("face detections", image) 

            (x, y, w, h) = face_utils.rect_to_bb(rectangle)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            cv2.putText(image, "face #{}".format(i + 1), (x - 10, y - 10),
	        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
                
                # show the output image with the facial landmarks
                if receive == self.keyl:
                    cv2.imshow("face landmarks", image)

        cv2.waitKey(0)


    def showlandmark(self):

        image = self.read
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detect = self.detection(gray, 1)
        img_id = self.count

        while True:

            img_id += 1
            pressedKey = cv2.waitKey(1) & 0xFF

            for (i, rectangle) in enumerate(detect):
                
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = self.prediction(gray, rectangle)
                shape = face_utils.shape_to_np(shape)

                (x, y, w, h) = face_utils.rect_to_bb(rectangle)

                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw them on the image
                for (x, y) in shape:
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
                 
            if pressedKey == ord(self.keyl):
                cv2.imwrite(self.write +str(img_id)+".jpg", image)
            elif pressedKey == ord(self.keyq):
                break

            # show the output image with the plot face detections
            cv2.imshow("Output", image)

        cv2.destroyAllWindows()
        cv2.waitKey(0)


    def detectface(self):

        cap = self.read
        img_id = self.count

        while True:
            
            # load the input image and convert it to grayscale
            _, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale image
            detect = self.detection(gray, 0)
            pressedKey = cv2.waitKey(1) & 0xFF

            # loop over the face detections
            for (i, rectangle) in enumerate(detect):
                
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = self.prediction(gray, rectangle)
                shape = face_utils.shape_to_np(shape)

                if pressedKey == ord(self.keyd):
                    cv2.imwrite(self.write +str(img_id)+".jpg", frame)
                img_id += 1

                (x, y, w, h) = face_utils.rect_to_bb(rectangle)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # show the face number
                cv2.putText(frame, "face #{}".format(i + 1), (x - 10, y - 10),
	            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw them on the image
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            if pressedKey == ord(self.keyl):
                cv2.imwrite(self.write +str(img_id )+".jpg", frame)
                img_id += 1

            elif pressedKey == ord(self.keyq):
                break

            # show the output image with the face detections
            cv2.imshow("Output", frame)
         
        cv2.destroyAllWindows()
        cap.release()
        

    def measure_model_error(self):

        # requires: the model and xml path.
        # It measures the error of the model on the given
        # xml file of annotations.
        error1 = dlib.test_shape_predictor(self.test, self.newmodel)
        error2 = dlib.test_shape_predictor(self.test, self.oldmodel)
        print("error of the model: {} is {}".format(self.newmodel, error1))
        print("error of the model: {} is {}".format(self.oldmodel, error2))

myfunction.close()


          



 
    


