from setuptools import setup, find_packages

setup(
    name='bk_comms',
    version='1.0',
    description='A useful module',
    author='Man Foo',
    author_email='foomail@foo.com',
    packages=find_packages(),  #same as name
    # entry_points={
    #     'console_scripts': [
    #         'bk_comms_main = bk_comms.main:main',
    #     ]
    # }
)
