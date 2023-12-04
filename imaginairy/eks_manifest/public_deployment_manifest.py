
def get_public_deployment_manifest(sqs_queue_url):
  public_deployment_manifest = {
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {
        "name": "public-pod-deployment",
        "labels": {
            "app": "public-pod",
        },
    },
    "spec": {
        "replicas": 1,  # Start with one replica, HPA will adjust as needed
        "selector": {
            "matchLabels": {
                "app": "public-pod",
            },
        },
        "template": {
            "metadata": {
                "labels": {
                    "app": "public-pod",
                },
            },
            "spec": {
                "containers": [
                    {
                        "name": "public-container",
                        "image": "your-public-container-image",  # Replace with your actual image
                        "ports": [
                            {
                                "containerPort": 80,  # Adjust if your application listens on a different port
                            },
                        ],
                        "env": [
                            {
                                "name": "SQS_QUEUE_URL",
                                "value": sqs_queue_url,
                            },
                        ],
                    },
                ],
                "affinity": {
                    "nodeAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": {
                            "nodeSelectorTerms": [
                                {
                                    "matchExpressions": [
                                        {
                                            "key": "kubernetes.io/role",
                                            "operator": "In",
                                            "values": ["frontend"],
                                        },
                                    ],
                                },
                            ],
                        },
                    },
                },
            },
        },
    },
  }
  return public_deployment_manifest