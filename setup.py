from setuptools import setup, find_packages

setup(
    name="paragem",
    version="1.0",
    description="A useful module",
    author="Behzad Karkaria",
    author_email="behzad.karkaria@gmail.com",
    packages=find_packages(),  # same as name
    # entry_points={
    #     'console_scripts': [
    #         'bk_comms_main = bk_comms.main:main',
    #     ]
    # }
)
