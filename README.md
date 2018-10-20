Ranking Reviews Helpfulness
==============================

This project aims to develop a machine learning approach to automatically rank reviews by their helpfulness

Project Organization
------------

   
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    |
    ├── docs               <- The projects documentation
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         and a short  description, e.g. 1.0-pre processing`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── preprocess_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    |   |   └── features_selection.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── train_and_predict.py
    │   │   └── ensemble_and_stacking.py
    │   │   └── evaluation.py
    │   │
    │
    

--------

Flow Chart
------------
<img src="docs/flow_chart.jpg" alt="flow_chart" align="center" width="700px"/>



Documentation
------------
See `./Doc/Kaggle_CrowdFlower_ChenglongChen.pdf` for documentation.

* <a href='https://drive.google.com/file/d/1ugqPiowyRqqIluPHLx3DdNd0EqPdrFHF/view?usp=sharing'>Ranking reviews by their helpfulness</a><br>

Instructions
------------
The project is depended on the following
<a href='https://bitbucket.org/talazaria/ranking_reviews_helpfulness/src/master/requirements.txt'>libraries</a><br>
Installation can be done by:
```commandline
pip install requierments.txt
```
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

