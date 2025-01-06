import boto3
import sagemaker
import pandas as pd
import numpy as np
import sagemaker.amazon.common as smac
from sagemaker import image_uris
from sagemaker.estimator import Estimator

role = "arn:aws:iam::340752796889:role/service-role/AmazonSageMakerExecutionRoleForBedrockMarketplace_I6RV1IGZSFC"

region = boto3.Session().region_name

sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()

s3 = boto3.client("s3")

filename = "wdc.s3"
s3.download_file("sagemaker-us-east-1-340752796889", "breast_cancer.csv", filename)

data = pd.read_csv(filename, header=None)

data.columns = [
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_main",
    "area_main",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

data.to_csv("data.csv", sep=",", index=False)

print(data.shape)
print(data.head())
print(data.describe())

rand_split = np.random.rand(len(data))
train_list = rand_split < 0.8
val_list = (rand_split >= 0.8) & (rand_split < 0.9)
test_split = rand_split >= 0.9

data_train = data[train_list]
data_val = data[val_list]
data_test = data[test_split]

train_y = ((data_train.iloc[:, 1] == "M") + 0).to_numpy()
train_x = data_train.iloc[:, 2:].to_numpy()

test_y = ((data_test.iloc[:, 1] == "M") + 0).to_numpy()
test_x = data_test.iloc[:, 2:].to_numpy()

train_data_path = f"s3://{bucket}/data/train"
validation_data_path = f"s3://{bucket}/data/validation"
sagemaker_session.upload_data(path="data.csv", bucket=bucket, key_prefix="data/train")

container = image_uris.retrieve(region=region, framework="linear-learner")

linear_learner = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    output_path=f"s3://{bucket}/output",
    sagemaker_session=sagemaker_session,
)

linear_learner.set_hyperparameters(
    predictor_type="binary_classifier",
    mini_batch_size=100,
    epochs=10,
    feature_dim=11,
)

train_data = sagemaker.inputs.TrainingInput(
    s3_data=train_data_path, content_type="csv"
)
validation_data = sagemaker.inputs.TrainingInput(
    s3_data=validation_data_path, content_type="csv"
)

linear_learner.fit(
    {"train": train_data, "validation": validation_data}, wait=True
)

