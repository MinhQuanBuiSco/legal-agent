import os

import yaml


class ConfigLoader:
    def __init__(self, config_file=None):
        # Determine the config file path dynamically
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
        default_config_file = os.path.join(base_dir, "../configs/config.yaml")  # Resolve absolute path

        # Allow custom config files (e.g., for dev/prod environments)
        self.config_file = config_file if config_file else default_config_file

        self.config = self.load_config()

    def load_config(self):
        """Load YAML configuration."""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file {self.config_file} not found.")

        with open(self.config_file, "r") as file:
            return yaml.safe_load(file)

    def get(self, key, default=None):
        """Retrieve a configuration value using dot notation (e.g., 'database.host')."""
        keys = key.split(".")
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value


config = ConfigLoader().config
