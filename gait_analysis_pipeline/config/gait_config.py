import yaml
import importlib


def load_pipeline_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Convert joint_enum string back to Python class
    module = importlib.import_module(config["joint_enum"]["module"])
    joint_enum_class = getattr(module, config["joint_enum"]["class"])

    config["joint_enum"] = joint_enum_class

    return config
