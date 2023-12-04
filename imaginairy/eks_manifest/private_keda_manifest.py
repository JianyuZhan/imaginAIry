
def get_private_keda_manifest(queue_url, region):
  sqs_scaled_object_manifest = {
      "apiVersion": "keda.sh/v1alpha1",
      "kind": "ScaledObject",
      "metadata": {
          "name": "sqs-scaledobject",
          "namespace": "default",  # Replace with the namespace of your choice
      },
      "spec": {
          "scaleTargetRef": {
              "kind": "Deployment",
              "name": "private-pod-deployment",  # This should match the name of your deployment
          },
          "minReplicaCount": 1,
          "maxReplicaCount": 10,
          "cooldownPeriod": 300,
          "triggers": [
              {
                  "type": "aws-sqs-queue",
                  "metadata": {
                      "queueURL": queue_url,
                      "queueLength": "5",
                      "awsRegion": region,
                      "identityOwner": "operator"
                  }
              }
          ]
      }
  }
  return sqs_scaled_object_manifest
