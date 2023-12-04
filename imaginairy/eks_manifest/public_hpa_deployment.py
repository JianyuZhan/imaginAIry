def get_public_hpa_manifest():
  public_hpa_manifest = {
    "apiVersion": "autoscaling/v2beta2",
    "kind": "HorizontalPodAutoscaler",
    "metadata": {
        "name": "public-pod-hpa",
    },
    "spec": {
        "scaleTargetRef": {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "name": "public-pod-deployment",
        },
        "minReplicas": 1,
        "maxReplicas": 10,
        "metrics": [
            {
                "type": "Resource",
                "resource": {
                    "name": "cpu",
                    "target": {
                        "type": "Utilization",
                        "averageUtilization": 50,  # Target CPU utilization percentage to scale up
                    },
                },
            },
        ],
    },
  }
  return public_hpa_manifest
