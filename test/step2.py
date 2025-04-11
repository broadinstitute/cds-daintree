
import os
import json
import sys

next_output_path = sys.argv[1]
rand_value = sys.argv[2]

with open(f"test_output/step2-completed-{rand_value}", "wt") as fd:
    fd.write("done")

with open(next_output_path, "wt") as fd:
    fd.write(json.dumps({    "type": "done",
        "args": []}))
    