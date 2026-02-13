import json
import os
from click.testing import CliRunner
from daintree_runner.main import cli


def test_create_sparkles_workflow_with_python_path(tmpdir):
    # Create a directory with some python files
    python_dir = tmpdir.mkdir("my_transforms")
    python_dir.join("preprocess.py").write("def transform(df): return df")
    python_dir.join("utils.py").write("def helper(): pass")

    # Create a mock taiga token
    taiga_token = tmpdir.join(".taiga-token")
    taiga_token.write("fake-token")

    # Change to tmpdir so _find_taiga_token finds our mock token
    old_cwd = os.getcwd()
    os.chdir(str(tmpdir))

    try:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(cli, [
            "create-sparkles-workflow",
            "--config", "model-map.json",
            "--python-path", str(python_dir),
        ])

        assert result.exit_code == 0, f"Command failed: {result.output}"

        workflow = json.loads(result.output)

        # Check that python files are in paths_to_localize
        paths_to_localize = workflow["paths_to_localize"]
        python_file_destinations = [
            p["dst"] for p in paths_to_localize
            if p["dst"].startswith("extra_python_files/")
        ]
        assert "extra_python_files/preprocess.py" in python_file_destinations
        assert "extra_python_files/utils.py" in python_file_destinations

        # Check that prepare command has --python-path extra_python_files
        prepare_command = workflow["steps"][0]["command"]
        assert "--python-path" in prepare_command
        python_path_idx = prepare_command.index("--python-path")
        assert prepare_command[python_path_idx + 1] == "extra_python_files"

        # Check that fit-model command has --python-path extra_python_files
        fit_model_command = workflow["steps"][1]["command"]
        assert "--python-path" in fit_model_command
        python_path_idx = fit_model_command.index("--python-path")
        assert fit_model_command[python_path_idx + 1] == "extra_python_files"

    finally:
        os.chdir(old_cwd)


def test_create_sparkles_workflow_without_python_path(tmpdir):
    # Create a mock taiga token
    taiga_token = tmpdir.join(".taiga-token")
    taiga_token.write("fake-token")

    old_cwd = os.getcwd()
    os.chdir(str(tmpdir))

    try:
        runner = CliRunner()
        result = runner.invoke(cli, [
            "create-sparkles-workflow",
            "--config", "model-map.json"
        ])

        assert result.exit_code == 0, f"Command failed: {result.output}"

        workflow = json.loads(result.output)

        # Check that no extra_python_files paths are in paths_to_localize
        paths_to_localize = workflow["paths_to_localize"]
        python_file_destinations = [
            p["dst"] for p in paths_to_localize
            if "extra_python_files" in p["dst"]
        ]
        assert len(python_file_destinations) == 0

        # Check that prepare command does NOT have --python-path
        prepare_command = workflow["steps"][0]["command"]
        assert "--python-path" not in prepare_command

        # Check that fit-model command does NOT have --python-path
        fit_model_command = workflow["steps"][1]["command"]
        assert "--python-path" not in fit_model_command

    finally:
        os.chdir(old_cwd)
