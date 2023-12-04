from aws_cdk import (
    aws_efs as efs,
    aws_apigateway as apigateway,
    aws_lambda as lambda_,
    aws_sqs as sqs,
    aws_eks as eks,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_s3 as s3,
    aws_s3_notifications as s3_nodify,
    aws_datasync as datasync,
    aws_karpenter as karpenter,
    core,
)
from eks_manifest import (
    get_public_deployment_manifest,
    get_public_hpa_manifest,
    get_private_deployment_manifest,
    get_private_keda_manifest, 
    get_efs_pv_manifest,
    get_efs_pvc_manifest,
)

class ImaginairyStack(core.Stack):
    def __init__(self, scope: core.Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        model_bucket_name = core.CfnParameter(self, "ModelS3Bucket",
                                              type="String",
                                              description="The name of the S3 bucket to sync with EFS")
        
        # Create a VPC with both public and private subnets
        vpc = ec2.Vpc(self, "ImaginairyVPC",
                      max_azs=3,
                      subnet_configuration=[
                          ec2.SubnetConfiguration(
                              name="PublicSubnet",
                              subnet_type=ec2.SubnetType.PUBLIC,
                              cidr_mask=24
                          ),
                          ec2.SubnetConfiguration(
                              name="PrivateSubnet",
                              subnet_type=ec2.SubnetType.PRIVATE_WITH_NAT,
                              cidr_mask=24
                          )
                      ])

        # Create an EFS FileSystem
        efs_filesystem = self.setup_efs(vpc)

        # Create an SQS queue
        queue = sqs.Queue(self, "ImaginairyQueue",
                          visibility_timeout=core.Duration.seconds(300))

        # Create a Lambda function to consume requests and write to SQS
        process_lambda = self.setup_request_comsumer_lambda(queue)

        # Create an API Gateway to accept requests
        self.setup_api_gateway(process_lambda)

        # Create the EKS cluster
        self.setup_eks_cluster(vpc, efs_filesystem, queue)

        # Setup DataSync to auto-sync S3 models to EFS, which could be mounted by the runtime pods
        self.setup_s3_datasync_to_efs(model_bucket_name.value_as_string, efs_filesystem, vpc)

    def setup_eks_cluster(self, vpc, efs_filesystem, sqs_queue):
        # Create an EKS cluster
        cluster_role = iam.Role(self, "ImaginairyClusterRole", assumed_by=iam.CompositePrincipal(
                                iam.ServicePrincipal("eks.amazonaws.com"),
                                iam.ServicePrincipal("ec2.amazonaws.com")
                                ))
        # This policy list is just an example, adjust the policies according to your actual requirements
        cluster_policy_statements = [
            iam.PolicyStatement(
                actions=["ecr:GetAuthorizationToken", "ecr:BatchCheckLayerAvailability",
                         "ecr:GetDownloadUrlForLayer", "ecr:GetRepositoryPolicy", "ecr:DescribeRepositories",
                         "ecr:ListImages", "ecr:DescribeImages", "ecr:BatchGetImage", "logs:CreateLogStream",
                         "logs:PutLogEvents", "s3:GetObject", "s3:PutObject", "eks:DescribeCluster"],
                resources=['*'],  # Scope down to the specific resources as necessary
            )
        ]
        for statement in cluster_policy_statements:
            cluster_role.add_to_policy(statement)

        eks_cluster = eks.Cluster(self, "ImaginairyEksCluster",
                                  vpc=vpc,
                                  default_capacity=0, # manage capacity using Karpenter
                                  version=eks.KubernetesVersion.V1_25,
                                  masters_role=cluster_role)
        
        # Use Karpenter to provision nodes in public subnets for pods consuming requests from SQS
        karpenter.Provisioner(self, "KarpenterPublicProvisioner",
                              cluster=eks_cluster,
                              subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC))

        # Use Karpenter to provision nodes in private subnets for backend pods, without public internet access
        karpenter.Provisioner(self, "KarpenterPrivateProvisioner",
                              cluster=eks_cluster,
                              subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_NAT))
        
        # Apply the manifests to the EKS cluster
        eks_cluster.add_manifest("PublicDeployment", get_public_deployment_manifest(sqs_queue.queue_url))
        eks_cluster.add_manifest("PublicHpa", get_public_hpa_manifest())
        eks_cluster.add_manifest("PrivateDeployment", get_private_deployment_manifest(sqs_queue.queue_url))
        eks_cluster.add_manifest("PrivateKeda", get_private_keda_manifest(sqs_queue.queue_url))
        eks_cluster.add_manifest("EfsPv", get_efs_pv_manifest(efs_filesystem.file_system_id))
        eks_cluster.add_manifest("EfsPvc", get_efs_pvc_manifest())

         # Output the EKS cluster name
        core.CfnOutput(self, "EksClusterName", value=eks_cluster.cluster_name)

    def setup_api_gateway(self, process_lambda):
        # Create the API Gateway to accept requests
        api = apigateway.RestApi(self, "ImaginairyAPI",
                                 rest_api_name="Imaginairy Service API",
                                 description="This service serves different image GenAI runtimes.")

        # Setup Lambda as an API Gateway backend
        integration = apigateway.LambdaIntegration(process_lambda,
                                                   request_templates={"application/json": '{ "statusCode": "200" }'})

        # Define the API Gateway method for receiving POST requests
        api.root.add_method("POST", integration)

        return api
    
    def setup_request_comsumer_lambda(self, sqs_queue):
        # Define an IAM role for the Lambda function
        lambda_role = iam.Role(self, "ImaginairyLambdaRole", assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"))
        # Assuming the Lambda needs to access Amazon S3, CloudWatch Logs, and Amazon EFS
        lambda_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name('service-role/AWSLambdaBasicExecutionRole')
        )
        lambda_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name('service-role/AWSLambdaVPCAccessExecutionRole')
        )
        lambda_role.add_to_policy(iam.PolicyStatement(
            resources=['*'],
            actions=['s3:GetObject', 's3:PutObject']  # Specify more fine-grained permissions as required
        ))

        # Create the Lambda function to process requests and write to SQS
        lambda_function = lambda_.Function(self, "ImaginairyLambda",
                                           runtime=lambda_.Runtime.PYTHON_3_10,
                                           handler="request_consumer.lambda_handler",
                                           code=lambda_.Code.asset('../src/lambda'),
                                           role=lambda_role,
                                           environment={
                                              # Pass the SQS queue URL as an environment variable
                                              'SQS_QUEUE_URL': sqs_queue.queue_url 
                                            })
        
                
        # Grant the Lambda function permission to write to the SQS queue
        sqs_queue.grant_send_messages(lambda_function)

        return lambda_function
    
    def setup_efs(self, vpc):
        # Create a security group for the EFS mount targets and EKS nodes
        efs_sg = ec2.SecurityGroup(self, "EfsSecurityGroup", vpc=vpc)
        efs_sg.add_ingress_rule(
            peer=efs_sg,
            connection=ec2.Port.tcp(2049),
            description="Allow NFS access for EFS"
        )
        
        # Define the EFS FileSystem
        efs_filesystem = efs.FileSystem(self, "ImaginairyEFS",
                                        vpc=vpc,
                                        lifecycle_policy=efs.LifecyclePolicy.AFTER_14_DAYS,  # Transition to IA after 14 days
                                        performance_mode=efs.PerformanceMode.GENERAL_PURPOSE,
                                        throughput_mode=efs.ThroughputMode.BURSTING,
                                        removal_policy=core.RemovalPolicy.DESTROY,
                                        vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_NAT),
                                        security_group=efs_sg)
        return efs_filesystem
    
    def setup_s3_datasync_to_efs(self, bucket_name, efs, vpc):
        datasync_s3_role = self.setup_datasync_role()
        bucket = s3.Bucket.from_bucket_name(self, "ModelS3Bucket", bucket_name)

        # Create a S3 bucket location for DataSync
        s3_location = datasync.CfnLocationS3(self, "S3Location",
                                             s3_bucket_arn=f"arn:aws:s3:::{bucket_name}",
                                             s3_storage_class="STANDARD",
                                             s3_config=datasync.CfnLocationS3.S3ConfigProperty(
                                                 bucket_access_role_arn=datasync_s3_role.role_arn
                                            ))

        # Create an EFS location for DataSync
        efs_location = datasync.CfnLocationEFS(self, "EfsLocation",
                                               efs_filesystem_arn=efs.file_system_arn,
                                               ec2_config=datasync.CfnLocationEFS.Ec2ConfigProperty(
                                                   security_group_arns=[efs.security_group.security_group_arn],
                                                   subnet_arn=vpc.private_subnets[0].subnet_arn
                                               ))

        # DataSync task to sync from S3 to EFS
        datasync_task = datasync.CfnTask(self, "DataSyncTask",
                                         source_location_arn=s3_location.attr_location_arn,
                                         destination_location_arn=efs_location.attr_location_arn,
                                         name="S3-to-EFS-DataSync",
                                         cloud_watch_log_group_arn="",
                                         )

        # Use a Lambda function triggered by S3 event to start a sync task
        datasync_start_lambda = lambda_.Function(
            self, 'DataSyncStartLambda',
            runtime=lambda_.Runtime.PYTHON_3_8,
            handler='start_datasync.lambda_handler',
            code=lambda_.Code.from_asset('../src/lambda'), 
            environment={
                'DATASYNC_TASK_ARN': datasync_task.attr_task_arn
            }
        )

        # Grant the Lambda function permissions to start DataSync tasks
        datasync_start_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=['datasync:StartTaskExecution'],
            resources=[datasync_task.attr_task_arn]
        ))

        # Configures the S3 bucket to notify Lambda function whenever a new object is added to the bucket.
        # The Lambda function will start the DataSync task to sync the new object to EFS.
        notification = s3_nodify.LambdaDestination(datasync_start_lambda)
        bucket.add_event_notification(s3.EventType.OBJECT_CREATED, notification)

        # Output the DataSync Task ARN
        core.CfnOutput(self, "DataSyncTaskArn", value=datasync_task.attr_task_arn)

    def setup_datasync_role(self, s3_bucket):
        # Create a new IAM role for DataSync to access S3
        datasync_s3_role = iam.Role(self, "DataSyncS3AccessRole",
                                    assumed_by=iam.ServicePrincipal("datasync.amazonaws.com"),
                                    description="Role for DataSync to access specific S3 bucket")

        # Define the policy allowing DataSync to list the bucket and read/write objects
        datasync_s3_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:ListBucket", 
                     "s3:GetBucketLocation", 
                     "s3:ListBucketMultipartUploads",
                     "s3:GetObject",
                     "s3:PutObject",
                     "s3:DeleteObject",
                     "s3:AbortMultipartUpload"],
            resources=[s3_bucket.bucket_arn, s3_bucket.arn_for_objects("*")],
        ))
        return datasync_s3_role