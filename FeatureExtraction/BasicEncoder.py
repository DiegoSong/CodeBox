from category_encoders import *

"""
category_encoders中包含
Backward Difference Contrast [2][3]
BaseN [6]
Binary [5]
Hashing [1]
Helmert Contrast [2][3]
LeaveOneOut [4]
Ordinal [2][3]
One-Hot [2][3]
Polynomial Contrast [2][3]
Sum Contrast [2][3]
Target Encoding [7]
Weight of Evidence [8]

Example
-------
import category_encoders as ce

encoder = ce.BackwardDifferenceEncoder(cols=[...])
encoder = ce.BinaryEncoder(cols=[...])
encoder = ce.HashingEncoder(cols=[...])
encoder = ce.HelmertEncoder(cols=[...])
encoder = ce.OneHotEncoder(cols=[...])
encoder = ce.OrdinalEncoder(cols=[...])
encoder = ce.SumEncoder(cols=[...])
encoder = ce.PolynomialEncoder(cols=[...])
encoder = ce.BaseNEncoder(cols=[...])
encoder = ce.TargetEncoder(cols=[...])
encoder = ce.LeaveOneOutEncoder(cols=[...])

encoder.fit(X, y)
X_cleaned = encoder.transform(X_dirty)


References
-------
.. [1] Kilian Weinberger; Anirban Dasgupta; John Langford; Alex Smola; Josh Attenberg (2009). Feature Hashing for 
        Large Scale Multitask Learning. Proc. ICML.
.. [2] Contrast Coding Systems for categorical variables. UCLA: Statistical Consulting Group. from 
        https://stats.idre.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/.
.. [3] Gregory Carey (2003). Coding Categorical Variables. from 
        http://psych.colorado.edu/~carey/Courses/PSYC5741/handouts/Coding%20Categorical%20Variables%202006-03-03.pdf
.. [4] Strategies to encode categorical variables with many categories. from 
        https://www.kaggle.com/c/caterpillar-tube-pricing/discussion/15748#143154.
.. [5] Beyond One-Hot: an exploration of categorical variables. from 
        http://www.willmcginnis.com/2015/11/29/beyond-one-hot-an-exploration-of-categorical-variables/
.. [6] BaseN Encoding and Grid Search in categorical variables. from 
        http://www.willmcginnis.com/2016/12/18/basen-encoding-grid-search-category_encoders/
.. [7] A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems.from 
        https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
.. [8] Weight of Evidence (WOE) and Information Value Explained. from 
        https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
"""
