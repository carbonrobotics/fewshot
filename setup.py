from setuptools import setup

if __name__ == "__main__":
    setup(
        name="fewshot",
        version="1.0.0",
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
