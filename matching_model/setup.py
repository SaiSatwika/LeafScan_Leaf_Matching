from setuptools import setup, find_packages

setup(
    name="leaf_matching",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "opencv-python",
        "scikit-learn",
        "torch",
        "torchvision",
        "joblib",
        "Pillow"
    ],
)