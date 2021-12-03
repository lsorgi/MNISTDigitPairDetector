from setuptools import setup, find_packages

REQUIRED = [
    "numpy==1.20.1",
    "opencv-python==4.5.1.48",
    "opencv-contrib-python==4.5.1.48",
    "albumentations==1.0.3",
    "matplotlib>=3.3.4<3.4",
    "tensorboard==2.6.0",
    "torchsummary==1.5.1",
    "torch==1.9.0",
    "torchvision==0.10.0",
    "pytest==6.2.2",
    "tqdm>=4.6<4.7",
    "dataclasses-json>=0.5<0.6"
]

setup(name='digit_pair_detector',
      version='0.1.0',
      author='Lorenzo Sorgi',
      author_email='lorenzosorgi77@gmail.com',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      include_package_data=False,
      zip_safe=False,
      install_requires=REQUIRED,
      python_requires='>=3.7'
)
