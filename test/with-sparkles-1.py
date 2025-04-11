import os
import json
import sys

next_output_path = sys.argv[1]
rand_value = sys.argv[2]

with open(f"test_output/1-{rand_value}", "wt") as fd:
    fd.write("done")

with open(next_output_path, "wt") as fd:
    fd.write(json.dumps({"type": "execute",
        "sparkles_command": ["python", "with-sparkles-worker.py"],
        "sparkles_uploads": ["with-sparkles-worker.py"],
        "args": ["python", "with-sparkles-2.py", "{next_step_filename}", rand_value]}))
    
