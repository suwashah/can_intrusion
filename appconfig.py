import json


class Config:
    def __init__(self, environment):
        config_file_path = f"Environment/{environment}_config.json"
        with open(config_file_path, 'r') as file:
            self.config_data = json.load(file)

    def getconfig_from_key(self, key):
        return self.config_data.get(key, None)

    def getconfig_from_keys(self, keys):
        current_level = self.config_data
        for key in keys:
            if key in current_level:
                current_level = current_level[key]
            else:
                return None
        return current_level
