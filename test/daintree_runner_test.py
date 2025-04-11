

import subprocess
import os
import random
import string 
runner_executable = "../daintree_runner"

def test_run_job():
    rand_value = ''.join([random.choice(string.ascii_letters) for _ in range(10)])
    breadcrumbs = [f"test_output/step1-completed-{rand_value}", f"test_output/step2-completed-{rand_value}"]

    os.makedirs("test_output", exist_ok=True)

    subprocess.check_call([runner_executable, "python", "python", "step1.py", "{next_step_filename}", rand_value])

    for fn in breadcrumbs:
        assert os.path.exists(fn)


def test_job_with_sparkles():
    rand_value = ''.join([random.choice(string.ascii_letters) for _ in range(10)])
    breadcrumbs = [f"test_output/1-{rand_value}", f"test_output/2-{rand_value}"]

    os.makedirs("test_output", exist_ok=True)

    subprocess.check_call([runner_executable, "python", "python", "with-sparkles-1.py", "{next_step_filename}", rand_value])

    for fn in breadcrumbs:
        assert os.path.exists(fn)


test_run_job()
test_job_with_sparkles()