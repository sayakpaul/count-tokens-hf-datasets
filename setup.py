import setuptools

NAME = "count_training_tokens"
VERSION = "0.0.1"
REQUIRED_PACKAGES = [
    "apache-beam[gcp]==2.34.0",
    "google-cloud-storage==2.2.1",
    "tokenizers==0.11.6",
    "transformers==4.17.0",
]

setuptools.setup(
    name=NAME,
    version=VERSION,
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
)
