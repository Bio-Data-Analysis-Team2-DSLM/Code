# Code
This project is part of DSLM Master class, Bio-Data analysis

Our topic will focus on the dataset provided in the article "Depresjon: a motor activity database of depression episodes in unipolar and bipolar patients" (https://dl.acm.org/doi/10.1145/3204949.3208125). The dataset is also available on Kaggle (https://www.kaggle.com/datasets/arashnic/the-depression-dataset) and includes 60 individuals with and without depression. It provides two types of information. The first is a 24-hour recording of the entire previous population's human activity per second through an actigraph smartwatch. The second information provided includes demographic and psychological characteristics, measurements from psychological tests, and the classification of depression into unipolar or bipolar (only for individuals suffering from it). The axes we would like to explore are as follows:

Creation of an explainable machine learning model for depression detection based on human activity measurements.
   
1) Use of 1-d CNN to actigraph signal to automatically extract features.

2) Extraction of other significant research-based features of actigraph signal.

3) Use of explainable models like XGBoost.

4) Comparison of results with simple statistical models (e.g., logistic regression).

Creation of an explainable machine learning model for depression type classification.
   
1) Data augmentation (evaluation and comparison of generated data and model response).

2) Use of explainable models.

The objective of this work will be to utilize biosensor data for potential depression detection and the ability to categorize it if present, using suitable metrics. Additionally, part of the work involves researching appropriate models that produce interpretable results and highlighting their contribution to medical diagnosis by psychiatrists. This work also includes the effort to handle a limited amount of data and possibly increase it, but with corresponding evaluation rather than indiscriminate use. Optionally, we will attempt to compare the models with simple statistical models to assess how much our models contribute to improved depression detection and classification.
