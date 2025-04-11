import os
import json
import sys

next_output_path = sys.argv[1]
rand_value = sys.argv[2]
sparkles_job = sys.argv[3]
sparkles_path = sys.argv[4]

with open(f"test_output/2-{rand_value}", "wt") as fd:
    fd.write(f"""
sparkles_job={sparkles_job}
sparkles_path={sparkles_path}
""")

with open(next_output_path, "wt") as fd:
    fd.write(json.dumps({"type": "done"}))
    