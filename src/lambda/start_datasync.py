import boto3
import os

def handler(event, context):
    # Create a boto3 client for DataSync
    client = boto3.client('datasync')

    # Get the DataSync Task ARN from the Lambda environment variable
    datasync_task_arn = os.environ['DATASYNC_TASK_ARN']

    # Start the DataSync task
    try:
        response = client.start_task_execution(TaskArn=datasync_task_arn)
        print(f"Started DataSync task: {response['TaskExecutionArn']}")
        return response['TaskExecutionArn']
    except Exception as e:
        print(e)
        raise e
