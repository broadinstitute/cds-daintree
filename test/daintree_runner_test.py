

import subprocess
import os
import random
import string 
runner_executable = "../daintree_runner"

def test_run_job():
    rand_value = ''.join([random.choice(string.ascii_letters) for _ in range(10)])
    breadcrumbs = [f"test_output/step1-completed-{rand_value}", f"test_output/step1-completed-{rand_value}"]

    os.makedirs("test_output", exist_ok=True)

    # clean up any past execution
    for fn in breadcrumbs:
        if os.path.exists(fn):
            os.unlink(fn)

    subprocess.check_call([runner_executable, "python", "python", "step1.py", rand_value])

    for fn in breadcrumbs:
        assert os.path.exists(fn)


test_run_job()