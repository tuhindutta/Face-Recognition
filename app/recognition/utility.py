from deepface import DeepFace
import os
import cv2
import numpy as np
from .detectFace import *


__SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CROPPED_IMG_DIR = os.path.join(__SCRIPT_DIR, "crops")


MODELS = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet"
]



# def simpleDetect(target):
#     face = FaceDetect()
#     match = MatchFace(REF_IMG_DIR, CROPPED_IMG_DIR, face, MODELS[1])
#     match.match(target)
#     return match.annotated_image


class MatchFace:

    def __init__(self, ref_img_dir, cropped_img_dir, model):
        self.ref_img_dir = ref_img_dir
        self.cropped_img_dir = cropped_img_dir
        self.face = FaceDetect()
        self.model = model

    def processTargetImage(self, target):
        crops = self.face.crop(target)
        _, coords = self.face.detect(target)
        records = {}
        idx = 0
        for face_img, face_coords in zip(crops.values(), coords.values()):
            records[str(idx)] = face_coords
            face_img = cv2.resize(face_img, (200,200))
            cv2.imwrite(os.path.join(self.cropped_img_dir, f"{idx}.jpg"), face_img)
            idx += 1

        self.coords = records

    def _delCropsDir(self):
        for file in os.listdir(self.cropped_img_dir):
            os.remove(os.path.join(self.cropped_img_dir, file))


    def match(self, target):
        self.processTargetImage(target)
        verification = {}
        for faces in os.listdir(self.cropped_img_dir):
            face_path = os.path.join(self.cropped_img_dir, faces)
            for ref_img in os.listdir(self.ref_img_dir):
                
                ref_img_path = os.path.join(self.ref_img_dir, ref_img)
                result = DeepFace.verify(
                    img1_path = face_path,
                    img2_path = ref_img_path,
                    model_name = self.model,
                    enforce_detection=True
                    )['verified']
                if result:
                    ref_img_name = ref_img.split('.')[0]
                    face_name = faces.split('.')[0]
                    verification[ref_img_name] = self.coords[face_name]

        self.matched = verification
        # print(verification)
        image = target.copy()
        for person, coords in verification.items():
            # person = list(verification.keys())[0]
            # coords = list(verification.values())[0]
            image = cv2.rectangle(image, coords[1], coords[0], color=(0,0,256),
                                                thickness=2)
            pos = tuple(np.array(coords[0])-20)
            image = cv2.putText(image, person, pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                                2, cv2.LINE_AA)
        self.annotated_image = image
        self._delCropsDir()
