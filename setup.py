from setuptools import setup, find_packages

setup(
    name='gym_metacar',
    version='0.0.1',
    author="Dr. Tristan Behrens (AI Guru)",
    author_email="tristan@ai-guru.de",
    description="OpenAI Gym wrapper for metacar.",
    long_description="OpenAI Gym wrapper for metacar.",
    long_description_content_type="text/markdown",
    url="https://github.com/AI-Guru/gym-metacar",
    install_requires=['gym', "pygame", "selenium"],
    packages=find_packages(),
    package_data={"gym_metacar": ["envs/resources/*.html"]}
)
