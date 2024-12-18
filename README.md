# face_recognition

Docker image: https://hub.docker.com/repository/docker/tkdutta/facerecognition/general

<br>

Perform the following steps to run the container locally:

1. Create the following 3 directories in your current working directory which are to be mounted to the container:

   I. reference_images : All images with proper title are stored which are to be identified.

   II. target_images : Images are stored which are the targets and the reference images are to be identified in these.

   III. processed_images : All identified images are stored here.

   <br>

3. Run the following docker command:

   
```bash
docker run -d \
  -p 5000:5000 \
  --name facerecognition \
  --mount type=bind,source=${PWD}\reference_images,destination=/app/reference_images \
  --mount type=bind,source=${PWD}\target_images,destination=/app/target_images \
  --mount type=bind,source=${PWD}\processed_images,destination=/app/processed_images \
  tkdutta/facerecognition:tag
```

<br>

<img width="500" alt="image" src="https://github.com/user-attachments/assets/02ae7f7e-d10c-4e75-ab4b-c99e035f8dfe" />
