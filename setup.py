from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='planebots',
      version='0.22',
      #     ext_modules = cythonize([os.path.join("vision","coverage","dwa.pyx"),
      #            os.path.join("vision","coverage","main.pyx")],annotate=True,language_level=3),
      install_requires=requirements,
      description='Python module containing all code for the graduate project of Arjan Vermeulen',
      # url='http://github.com/barrygwn/miabot_vision',
      author='Arjan Vermeulen',
      package_data={
          '': ['data', 'thesisplots'],
      },
      author_email='barrygwn@gmail.com',
      license='GNU v3.0',
      packages=['planebots', ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=False,
      zip_safe=False)
