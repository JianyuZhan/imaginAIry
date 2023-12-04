#!/usr/bin/env python3
import argparse
import os

import aws_cdk as cdk

from imaginairy.imaginairy_stack import ImaginAIryStack

# Setup and parse command-line argument.
parser = argparse.ArgumentParser(description='Deploy CDK Stack')
parser.add_argument('--region', type=str, default='us-east-1', help='AWS region')
parser.add_argument('--account', type=str, default='101405808103', help='AWS account ID')
args = parser.parse_args()

app = cdk.App()
env = cdk.Environment(account=args.account, region=args.region)
app.synth()
ImaginAIryStack(app, "ImaginAIryStack", env=env)

app.synth()
