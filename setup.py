import subprocess

from setuptools import setup

def get_git_version():
    try:
        version = subprocess.check_output(["git", "describe", "--tags"]).strip().decode("utf-8")
        return version
    except subprocess.CalledProcessError:
        return "0.0.0"  # Fallback version if the command fails

if __name__ == "__main__":
    setup(
        name="fewshot",
        version=get_git_version(),
        description="The Python package for few-shot learning",
        author="Zachary New",
        author_email="",
        url="https://github.com/ZachNew/fewshot-package",
        packages=[
            "fewshot",
            "fewshot.encoders",
            "fewshot.models",
        ],
    )
