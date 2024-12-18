from flask import Flask, render_template, request, send_from_directory
import os
import cv2
from pathlib import Path
from recognition.utility import *

# Directories
REF_IMG_DIR = "./reference_images"
TARGET_IMG_DIR = "./target_images"
PROCESSED_IMG_DIR = "./processed_images"
model = MODELS[1]

match = MatchFace(REF_IMG_DIR, CROPPED_IMG_DIR, model)

# Ensure the processed directory exists
Path(PROCESSED_IMG_DIR).mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    # Ensure the target directory exists
    if not os.path.exists(TARGET_IMG_DIR):
        return f"Error: Directory '{TARGET_IMG_DIR}' does not exist.", 500

    # List all images in the target directory
    images = [img for img in os.listdir(TARGET_IMG_DIR) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
    selected_image = None
    processed_image_path = None
    if request.method == "POST":
        # Get the selected image from the form
        selected_image = request.form.get("image")
        
        if selected_image:
            # Load the selected image using OpenCV
            input_image_path = os.path.join(TARGET_IMG_DIR, selected_image)
            image = cv2.imread(input_image_path)

            # Perform OpenCV manipulations here
            try:
                manipulated_image = opencv_manipulations(image, match)
            except:
                manipulated_image = image
                print("Could not find any face")

            # Save the manipulated image to the processed directory
            processed_image_path = os.path.join(PROCESSED_IMG_DIR, selected_image)
            cv2.imwrite(processed_image_path, manipulated_image)
    
    return render_template(
        "index.html", 
        images=images, 
        selected_image=selected_image, 
        processed_image=(selected_image if processed_image_path else None)
    )

@app.route("/images/<filename>")
def images(filename):
    # Serve the selected image from the target directory
    return send_from_directory(TARGET_IMG_DIR, filename)

@app.route("/processed/<filename>")
def processed(filename):
    # Serve the processed image from the processed directory
    return send_from_directory(PROCESSED_IMG_DIR, filename)




def opencv_manipulations(image, match:MatchFace):
    """
    Apply OpenCV manipulations to the image.
    Modify this function with your desired transformations.
    """

    # 
    match.match(image)

    image = match.annotated_image
    print(image.shape)
    height, width = image.shape[:2]

    width_target = 500
    fac = height/width_target
    height_target = int(height/fac)

    image = cv2.resize(image, (height_target, width_target))
    return image

if __name__ == "__main__":
    app.run(debug=True)
