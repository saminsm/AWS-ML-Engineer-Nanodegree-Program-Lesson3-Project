#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
serializeImageData: a lambda function for pull an image from S3 and returning serializing data
"""
import json
import boto3
import base64

s3 = boto3.resource('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']

    # Download the data from s3 to /tmp/image.png
    s3.Bucket(bucket).download_file(key, '/tmp/image.png')

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        
        "image_data": image_data,
        "s3_bucket": bucket,
        "s3_key": key,
        "inferences": []
        
    }




"""
lambdaClassifier: a lambda function for the classification of the image_data
"""

# Name of the deployed model
ENDPOINT = "image-classification-2021-12-31-18-49-17-889"
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['image_data'])

    # Instantiate a Predictor
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT, ContentType='application/x-image', Body=image)
    inferences = response['Body'].read().decode('utf-8')
    event["inferences"] = [float(x) for x in inferences[1:-1].split(',')]
    
    # We return the data back to the Step Function    
    return {
        'statusCode': 200,
        "image_data": event['image_data'],
        "s3_bucket": event['s3_bucket'],
        "s3_key": event['s3_key'],
        "inferences": event['inferences'],
        
    }


"""
thresLambda: a lambda function to pick up only the cases with confidence higher than or equal to some threshold
"""

import json


THRESHOLD = .93


def lambda_handler(event, context):
    
    # Grab the inferences from the event
    inferences = event['inferences']
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any (x >= THRESHOLD for x in inferences)
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        "image_data": event['image_data'],
        "s3_bucket": event['s3_bucket'],
        "s3_key": event['s3_key'],
        "inferences": event['inferences'],
        
    }

