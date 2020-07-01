import os
import sys

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

# TRAINING_KEY_ENV_NAME = "e1ceb0b98f0543de9694c3309060c564"
# SUBSCRIPTION_KEY_ENV_NAME = "e1ceb0b98f0543de9694c3309060c564"

PUBLISH_ITERATION_NAME = "classifyModel"

ENDPOINT = "https://westeurope.api.cognitive.microsoft.com"

# Add this directory to the path so that custom_vision_training_samples can be found
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "."))

IMAGES_FOLDER = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "images")



def find_or_train_project():
    # try:
    #     training_key = "e1ceb0b98f0543de9694c3309060c564" # os.environ[TRAINING_KEY_ENV_NAME]
    # except KeyError:
    #     raise SubscriptionKeyError("You need to set the {} env variable.".format(TRAINING_KEY_ENV_NAME))

    # Use the training API to find the SDK sample project created from the training example.
    # from custom_vision_training_samples import train_project, SAMPLE_PROJECT_NAME

    from msrest.authentication import ApiKeyCredentials
    credentials = ApiKeyCredentials(in_headers={"Training-key": "e1ceb0b98f0543de9694c3309060c564"})
    trainer = CustomVisionTrainingClient("https://westeurope.api.cognitive.microsoft.com",credentials)

    for proj in trainer.get_projects():
        if (proj.name == "Bill-ee"):
            return proj

    # Or, if not found, we will run the training example to create it.
    # return train_project(training_key)


def predict_project():
    
    from msrest.authentication import ApiKeyCredentials
    credentials = ApiKeyCredentials(in_headers={"Training-key": "e1ceb0b98f0543de9694c3309060c564"})

    trainer = CustomVisionTrainingClient("https://westeurope.api.cognitive.microsoft.com",credentials)
    project = trainer.get_projects()[0]
    # trainer.publish_iteration(project.id, "11e9a49f-7aaa-479f-9353-0063006d9901", PUBLISH_ITERATION_NAME, "/subscriptions/8924a908-5005-456c-9dcc-b674e71a34ca/resourceGroups/bill-ee-rg/providers/Microsoft.CognitiveServices/accounts/bill-ee-cog")

    
    credentials = ApiKeyCredentials(in_headers={"Prediction-key": "c1e60818aba9439284caf3557859c658"})
    predictor = CustomVisionPredictionClient("https://westeurope.api.cognitive.microsoft.com", credentials)

    # Find or train a new project to use for prediction.
    # project = find_or_train_project()
    

    with open(os.path.join(IMAGES_FOLDER, "Test", "test_image.jpg"), mode="rb") as test_data:
        testimgstream = test_data.read()
        results = predictor.classify_image(project.id, PUBLISH_ITERATION_NAME, testimgstream)

    # Display the results.
    for prediction in results.predictions:
        print("\t" + prediction.tag_name +
              ": {0:.2f}%".format(prediction.probability * 100))


if __name__ == "__main__":
    import sys, os.path
    sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
    # from samples.tools import execute_samples, SubscriptionKeyError
    # execute_samples(globals(), SUBSCRIPTION_KEY_ENV_NAME)
    predict_project()