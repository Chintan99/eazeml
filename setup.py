import setuptools
 
with open("README.md", "r") as fh:
    long_description = fh.read()
 
setuptools.setup(
    #Here is the module name.
    name="eazeml",
 
    #version of the module
    version="0.0.7",

    #license
    license='LICENSE.txt',
    
    #Name of Author
    author="Chintan Chitroda",
 
    #your Email address
    author_email="chintanchitroda47@gmail.com",
 
    #Small Description about module
    description="EazeML makes Task of Machine Learning and Data Science super easy.",
 
    long_description=long_description,
 
    #Specifying that we are using markdown file for description
    long_description_content_type="text/markdown",
 
    #Any link to reach this module, if you have any webpage or github profile
    url="https://github.com/Chintan99/eazeml",
    packages=['eazeml'],
 
    #classifiers like program is suitable for python3, just leave as it is.
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    install_requires=[
        'pandas',
        'numpy',
        'seaborn',
        'matplotlib',
        'plotly',
        'sklearn',
        'lightgbm',
        'textblob',
        'nltk',
        'tqdm',
        'flask',
        'wordcloud',
        'xgboost ',
        'wordcloud',
        'cufflinks',
        'catboost',

    ],
)
