from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import time
import os

def main():
    from dotenv import load_dotenv
    global training_client
    global custom_vision_project

    try:
        # Get Configuration Settings
        load_dotenv()
        training_endpoint = os.getenv('TrainingEndpoint')
        training_key = os.getenv('TrainingKey')
        project_id = os.getenv('ProjectID')

        # Authenticate a client for the training API
        credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
        training_client = CustomVisionTrainingClient(training_endpoint, credentials)

        # Get the Custom Vision project
        custom_vision_project = training_client.get_project(project_id)

        # Upload and tag images
        Upload_Images('more-training-images')

        # Train the model
        Train_Model()
        
    except Exception as ex:
        print(ex)

# Function that uploads images to the Custom Vision service
def Upload_Images(folder):
    print("Uploading images...")
    
    # Get the tags from the project
    tags = training_client.get_tags(custom_vision_project.id)
    for tag in tags:
        print(tag.name)
        # for each image in the folder, upload and tag it
        for image in os.listdir(os.path.join(folder,tag.name)):
            # Get the image data
            image_data = open(os.path.join(folder,tag.name,image), "rb").read()
            # Upload the image and tag it with the appropriate tag
            training_client.create_images_from_data(custom_vision_project.id, image_data, [tag.id])

def Train_Model():
    print("Training ...")
    # Train the project
    iteration = training_client.train_project(custom_vision_project.id)
    # Wait for training to complete
    while (iteration.status != "Completed"):
        # re-check the status every 5 seconds
        iteration = training_client.get_iteration(custom_vision_project.id, iteration.id)
        print (iteration.status, '...')
        time.sleep(5)
    print ("Model trained!")


if __name__ == "__main__":
    main()


