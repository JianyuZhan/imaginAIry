import json
import os

import boto3
from botocore.exceptions import ClientError

# Initialize a boto3 client for SQS
sqs_client = boto3.client('sqs')
queue_url = os.environ['SQS_QUEUE_URL']


def lambda_handler(event, context):
    try:
        # Process the incoming request data as needed
        # Here, we're assuming the incoming event is a dictionary that can be converted to JSON
        # You may need to adjust the processing logic based on your specific use case
        message_body = json.dumps(event)

        # Send the message to the SQS queue
        response = sqs_client.send_message(
            QueueUrl=queue_url,
            MessageBody=message_body
        )

        # Log the message ID and MD5
        print(f"Message ID: {response['MessageId']}")
        print(f"Message MD5: {response['MD5OfMessageBody']}")

        return {
            'statusCode': 200,
            'body': json.dumps('Message sent to SQS queue successfully!')
        }

    except ClientError as e:
        print(e.response['Error']['Message'])
        return {
            'statusCode': 500,
            'body': json.dumps('Error sending message to SQS queue.')
        }
