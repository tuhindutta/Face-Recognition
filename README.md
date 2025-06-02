# 🧠 Face Recognition Web App
This is a Flask-based web application that allows you to perform face verification using DeepFace on user-uploaded or preloaded target images, comparing them against reference images.

## 📁 Project Structure
```env
Face-Recognition/
├── app/
│   ├── recognition/
│   │   ├── detectFace.py
│   │   └── utility.py
│   ├── templates/
│   │   └── index.html
│   ├── main.py
│   └── requirements.txt
├── Dockerfile
├── README.md
```

## ⚙️ Features
- Detects faces from a selected image.
- Crops and compares faces against stored reference images using DeepFace.
- Annotates matched faces with labels on the processed image.
- Displays processed output on the webpage.

## ⭐ Models, Libraries, and Technologies Used
- DeepFace Model used: **Facenet**
- Python Libraries:
  - **Mediapipe** detecting faces in the input images and extract precise face coordinates, which are then utilized for cropping and further face recognition processing.
  - **DeepFace** for face recognition and verification
  - **OpenCV** for image processing, cropping, resizing, and annotation
  - **NumPy** for numerical operations on image coordinates
  - **Flask** for building the web application and handling HTTP requests
- Deployment & Environment:
   - Docker container based on Ubuntu 22.04
   - Python dependencies managed with pip via `requirements.txt`
  
## 🛠 Setup Instructions
1. Perform the following steps to run the container locally:
      1. Create the following 3 directories in your current working directory which are to be mounted to the container:
            - `reference_images/` : Contains labeled face images for identification.
            - `target_images/` : Upload target images here.
            - `processed_images/` : Auto-generated results with annotations.
      2. Run the following docker command (Docker image: tkdutta/facerecognition):
         ```bash
         docker run -d \
           -p 5000:5000 \
           --name facerecognition \
           --mount type=bind,source=${PWD}\reference_images,destination=/app/reference_images \
           --mount type=bind,source=${PWD}\target_images,destination=/app/target_images \
           --mount type=bind,source=${PWD}\processed_images,destination=/app/processed_images \
           tkdutta/facerecognition:tag
         ```

## 🧪 How It Works
1. User selects a target image.
2. FaceDetect class (in detectFace.py) detects and crops faces.
3. MatchFace class:
   1. Matches cropped faces with reference images using DeepFace.verify.
   2. Annotates matched faces on the image using OpenCV.
4. Final annotated image is resized and saved.

## 🌐 Accessing the App
Open your browser and navigate to:
```url
http://localhost:5000
```

## 📌 To Do (Suggested Enhancements)
- Add image upload capability.
- Use session IDs for handling multiple users.
- Allow dynamic model selection.
- Integrate drag-and-drop UI.

## 🚀 Demo:
<img width="500" alt="image" src="https://github.com/user-attachments/assets/02ae7f7e-d10c-4e75-ab4b-c99e035f8dfe" />
