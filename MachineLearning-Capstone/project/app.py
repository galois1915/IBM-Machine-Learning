'''
The main grading criteria will be:

- Uploaded your completed presentation in PDF format (2 pts)
- Completed the required Introduction slide (4 pt)
- Completed the required Exploratory Data Analysis slides (8 pts)
- Completed the required content-based recommender system using user profile and course genres slides (6 pts)
- Completed the required content-based recommender system using course similarity slides (6 pts)
- Completed the required content-based recommender system using user profile clustering slides (6 pts)
- Completed the required KNN-based collaborative filtering slide (6 pts)
- Completed the required NMF-based collaborative filtering slide (6 pts)
- Completed the required neural network embedding based collaborative filtering slide (6 pts)
- Completed the required collaborative filtering algorithms evaluation slides (6 pts)
- Completed the required Conclusion slide (6 pts)
- Applied your creativity to improve the presentation beyond the template (4 pts)
- Displayed any innovative insights (4 pts)
'''

import streamlit as st
import numpy as np

tab1, tab2 = st.tabs(["Introduction","EDA","K Means", "NMF", "NN"])
data = np.random.randn(10, 1)

tab1.subheader("A tab with a chart")
tab1.line_chart(data)

tab2.subheader("A tab with the data")
tab2.write(data)