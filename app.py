import numpy as np
import pandas as pd
import streamlit as st 
import pandas as pd
import numpy as np
from fastapi import FastAPI, Response, status
from pydantic import BaseModel
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('combined_data.csv')
movies = pd.read_csv('movies-clean.csv', index_col = 'movieId')

pivot_table = data.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(0)
content_based_info = pd.DataFrame(data.drop(columns = ['title', 'userId', 'timestamp', '(no genres listed)']).groupby(by = 'movieId').mean())


def show_recommendations(movieId):    
    
    
    def collaborative_cosine(movieId, data):    
        movie_indices = data.index   
        index = np.where(movie_indices == movieId)    
        movie_similarity_cosine = cosine_similarity(data)
        similarities = movie_similarity_cosine[index].ravel()    
        closest = np.argpartition(similarities, -11)[-11:]
        recommended_movie_indices = movie_indices[closest].tolist()    
        if movieId in recommended_movie_indices:
            recommended_movie_indices.remove(movieId)    
        recommended_movies = movies.loc[np.array(recommended_movie_indices),:].index.tolist()
        return recommended_movies
    
    
    result = []
    for i in range(0,len(movieId)):   
        result.extend(collaborative_cosine(movieId[i], pivot_table))
    result = list(dict.fromkeys(result))
    result = content_based_info.loc[result,:].sort_values(by = 'rating').index[0:10]
    result = movies.loc[result,'disptitle']
    
    return result    
def display_movies(arr): #displays names of movies given movieIds
    res = []
    for id in arr:       
        
        res.append(movies.loc[movies.index==id, 'disptitle'].iloc[0])
    return res

def cold_start():
    highly_rated = content_based_info.sort_values(by=['rating'], ascending = False).index.values[0:20]
    return display_movies(highly_rated)

def get_id(names):
    id = []
    for name in names:
        id.append(movies[movies['disptitle'] == name].index.tolist()[0])
    
    print(id)
    return id


def main():
    st.title("Movie recommendations")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> Movie recommender system</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    options = st.multiselect(
     'What movies do you like?',
     ['Forrest Gump (1994)','Shawshank Redemption, The (1994)','Pulp Fiction (1994)','Silence of the Lambs, The (1991)',
 'Matrix, The (1999)', 'Star Wars: Episode IV - A New Hope (1977)', 'Jurassic Park (1993)', 'Braveheart (1995)', 'Terminator 2: Judgment Day (1991)',
 "Schindler's List (1993)", 'Fight Club (1999)', 'Toy Story (1995)', 'Star Wars: Episode V - The Empire Strikes Back (1980)',
 'Usual Suspects, The (1995)', 'American Beauty (1999)', 'Seven (a.k.a. Se7en) (1995)', 'Independence Day (a.k.a. ID4) (1996)',
 'Apollo 13 (1995)', 'Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)',
 'Lord of the Rings: The Fellowship of the Ring, The (2001)'],
     )

    st.write('You selected:', options)
    movies_id_selected = get_id(options)
    print(options)    
    result=[]
    if st.button("Predict"):
        if(options == []):
            result = cold_start()
        else:
            result = show_recommendations(movies_id_selected)
        
    st.write(result)
    if st.button("About"):
        st.text("Recommendations generated using Item-based collaborative filtering")

if __name__=='__main__':
    main()
    
