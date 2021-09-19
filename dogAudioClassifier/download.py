# # azureml-core of version 1.0.72 or higher is required
# from azureml.core import Workspace, Dataset

# subscription_id = 'afc8afff-59cf-4a3f-85d3-6b742b7a5aa2'
# resource_group = 'resource'
# workspace_name = 'hackthenorth'

# workspace = Workspace(subscription_id, resource_group, workspace_name)

# dataset = Dataset.get_by_name(workspace, name='dog_audio')
# dataset.download(target_path='data/v2', overwrite=False)

import os
import shutil
import re
# Move files into correct folders
for f in os.listdir('data/v2'):
    if '.wav' in f:
        find = re.compile(r"^[^.]*")
        result = re.search(find, f)
        shutil.move('data/v2/'+f, 'data/v2/'+result.group(0)[-1])