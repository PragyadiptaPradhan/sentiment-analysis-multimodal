from sagemaker.pytorch import PyTorch 
from sagemaker.debugger import TensorBoardOutputConfig 


def start_training():
    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path="s3://multimodal-analysis-hrisi/tensorboard",
        container_local_output_path="/opt/ml/output/tensorboard"
    )

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="training",
        role="arn:aws:iam::490939613248:role/sentiment-analysis-execution-role",
        framework_version="2.5.1",
        py_version="py311",
        instance_count=1,
        instance_type="ml.g5.xlarge",
        hyperparameters={
            "batch-size": 32,
            "epochs": 25
        },
        tensorboard_config=tensorboard_config
    )

    # Start training
    estimator.fit({
        "training": "s3://multimodal-analysis-hrisi/dataset/train",
        "validation": "s3://multimodal-analysis-hrisi/dataset/dev",
        "test": "s3://multimodal-analysis-hrisi/dataset/test"
    })


if __name__ == "__main__":
    start_training()