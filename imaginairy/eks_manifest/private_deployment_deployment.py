def get_private_deployment_manifest():
    private_deployment_manifest = {
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {
        "name": "private-pod-deployment",
        "labels": {
            "app": "private-pod",
        },
    },
    "spec": {
        "replicas": 1,
        "selector": {
            "matchLabels": {
                "app": "private-pod",
            },
        },
        "template": {
            "metadata": {
                "labels": {
                    "app": "private-pod",
                },
            },
            "spec": {
                "containers": [
                    {
                        "name": "private-container",
                        "image": "your-private-container-image",  # Replace with your actual image
                        "volumeMounts": [
                            {
                                "name": "efs-volume",
                                "mountPath": "/mnt/efs",
                            },
                        ],
                    },
                ],
                "volumes": [
                    {
                        "name": "efs-volume",
                        "persistentVolumeClaim": {
                            "claimName": "efs-claim",
                        },
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
                                            "values": ["backend"],
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
return 

