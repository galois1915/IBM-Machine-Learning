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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Introduction",
                                        "EDA",
                                        "User profile and course genres", 
                                        "Course similarity",
                                        "Clustering",
                                        'Personalized Recimmender'])

## INTRODUCTION
tab1.subheader("A tab with a chart")
tab1.markdown('''
              ##### Overview project
              Overview
              ''')

## EDA
tab2.subheader("Exploratory data Analysis")
tab2.markdown('''
              ##### Showing the analysis
              BoW, etc
              ''')

## USER PROFILE AND COURSE GENRES
tab3.subheader('USER PROFILE AND COURSE GENRES')
tab3.markdown('''
              ##### Recommed the top best courses to users
              We have the **profile** of the users and the courses which are the users **enrolles**. 
              We will recommend the couress no seen based on his profile. The processed consist in the following steps:
              - Load the file with the info of enrollens, generate the dataframe with the recommendations for users.
              - Show the dataframe and add button if you want save the dataframe. Apply filter if you want.
              ''')

## COURSE SIMILARITY
tab4.subheader('Course similarity')
tab4.markdown('''
              ##### Recommed courses bases to especific course
              We have the description of each course, in this case will recommend courses bases in the
              enrollement of user. We upload the file with the enrollemnet and the output will be 
              recommend courses based on it.
              ''')

## CLUSTERING
tab5.subheader('Clustering')
tab5.markdown('''
              ##### Cluster users based in profile
              Using the profile of the users, we will cluster them using K-Means and PCA. Generate a dataframe 
              that have the labels for each user. With the new groups, we will recommend the top courses in the 
              group which the user belong. You can show the recommended course using a fataframe or can apply filters.
              In this case you also need to upload the file with the enrollements.

              > Keep in ming that some users may not be in  the upload file.
              ''')

## CLUSTERING
tab5.subheader('Personalized Recimmender')
tab5.markdown('''
              ##### Personalized Learning Recommender
              Primero mostrar un data frame con todos los cursos, si es podible un buscador usando palbras clave.
              Select the courses that you are enrollent or eudidedt, sellect the task:
              - recomendart basado en los cusrsos
              - basado en su perfil
              - basado bajo su cluster - aplicar pca como opcion
              - NN

              Mostrar dataframe mosrtando las recomendaciones, y tener en cuenta qyue cada metodo tiene sus parametros
              ''')