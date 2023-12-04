
def get_efs_pv_manifest(efs_id):
    efs_pv_manifest = {
      "apiVersion": "v1",
      "kind": "PersistentVolume",
      "metadata": {
          "name": "efs-pv",
      },
      "spec": {
          "capacity": {
              "storage": "5Gi"
          },
          "volumeMode": "Filesystem",
          "accessModes": [
              "ReadWriteMany"
          ],
          "persistentVolumeReclaimPolicy": "Retain",
          "storageClassName": "efs-sc",
          "csi": {
              "driver": "efs.csi.aws.com",
              "volumeHandle": efs_id # This will be the EFS FileSystem ID from CDK context
          }
      }
    }

    return efs_pv_manifest

def get_efs_pvc_manifest():
  efs_pvc_manifest = {
      "apiVersion": "v1",
      "kind": "PersistentVolumeClaim",
      "metadata": {
          "name": "efs-claim",
      },
      "spec": {
          "accessModes": [
              "ReadWriteMany"
          ],
          "storageClassName": "efs-sc",
          "resources": {
              "requests": {
                  "storage": "5Gi"
              }
          }
      }
  }
  return efs_pvc_manifest
