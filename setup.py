from setuptools import setup

setup(
    name='glom_pop',
    version='0.2',
    description='',
    url='https://github.com/mhturner/glom_pop',
    author='Max Turner',
    author_email='mhturner@stanford.edu',
    packages=['glom_pop'],
    install_requires=['numpy',
                      'h5py',
                      'scikit-image',
                      'colorcet',
                      'pandas',
                      'ipython',
                      'ipykernel',
                      'pyyaml',
                      'nibabel',
                      'seaborn',
                      ],
    include_package_data=True,
    zip_safe=False,
)
