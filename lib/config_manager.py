import json
import os
import shutil

def load_config():
   config_file = "config.json"
   user_config_file = "user_config.json"

   if not os.path.isfile(user_config_file):
       shutil.copy(config_file, user_config_file)

   with open(user_config_file, 'r') as user_file:
       user_config = json.load(user_file)

   with open(config_file, 'r') as config_file:
       config = json.load(config_file)

       # Merge the user configuration with the main configuration
       config = {**config, **user_config}

       missing_keys = set(config.keys()) - set(user_config.keys())

       if missing_keys:
           raise Exception(f"Missing keys in user_config.json: {missing_keys}")

           for key in missing_keys:
               user_config[key] = config[key]

           with open(user_config_file, 'w') as user_file:
               json.dump(user_config, user_file, indent=2)

       return user_config

def is_numeric(value):
   return isinstance(value, (int, float)) and (isinstance(value, float) or str(value).replace('.', '', 1).isdigit())

def get_config(key, default=None):
   config = load_config()

   if key not in config:
       config[key] = default
       update_config(key, default)
       load_config()  # reload the config after updating it

   # Check if the value is numeric and cast it to the appropriate type
   if is_numeric(config[key]):
       config[key] = int(config[key]) if float(config[key]).is_integer() else float(config[key])

   return config.get(key, default)

def update_config(key, value):
   config = load_config()

   # Cast the value to the appropriate type if it is numeric
   if is_numeric(value):
       value = int(value) if float(value).is_integer() else float(value)

   config[key] = value

   with open(user_config_file, 'w') as file:
       json.dump(config, file, indent=2)