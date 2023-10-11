"""All imports"""
import re
import logging
import pandas as pd
import numpy as np
import ipywidgets as widgets
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from IPython.display import display


logging.basicConfig(filename='errors.log', level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def log_errors(func):
    """
    Decorator para capturar erros e escrever no log
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as val_err:
            logging.error('Erro na função %s : %s', {func.__name__},  {val_err}, exc_info=True)
        except NameError as name_err:
            logging.error('Erro na função %s : %s', {func.__name__},  {name_err}, exc_info=True)
    return wrapper

# https://files.grouplens.org/datasets/movielens/ml-25m.zip
movies = pd.read_csv("movies.csv")

@log_errors
def clean_title(title):
    """
    Remove caracteres especiais de um título de filme.
    
    Args:
        title (str): Título do filme.
        
    Returns:
        str: Título do filme sem caracteres especiais.
    """
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title
movies["clean_title"] = movies["title"].apply(clean_title)
logging.info("Coluna 'clean_title' criada com sucesso")

#Mede a raridade de uma palavra em um conjunto de documentos.
vectorizer = TfidfVectorizer(ngram_range=(1,2))

tfidf = vectorizer.fit_transform(movies["clean_title"])

@log_errors
def search(title):
    """
    Encontra filmes semelhantes com base no título fornecido pelo usuário.
    
    Args:
        title (str): Título do filme inserido pelo usuário.
        
    Returns:
        pandas.DataFrame: Dataframe contendo os filmes semelhantes.
    """
    logging.info("Procurando filmes equivalentes no databese")
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]

    return results



@log_errors
def on_type(data):
    """ Atualiza a lista de filmes semelhantes quando o usuário digita um título de filme.
    Args:
        data (dict): Dados do evento de entrada interativa.
    """
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            movie_id = results.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))

@log_errors
def find_similar_movies(movie_id):
    """
    Encontra filmes semelhantes com base no ID do filme 
    fornecido pelo usuário e nas classificações dos usuários.
    
    Args:
        movie_id (int): ID do filme fornecido pelo usuário.
        
    Returns:
        pandas.DataFrame: Dataframe contendo os filmes semelhantes e suas informações.
    """
    logging.info("Procurando filmes similares de acordo com o rating    ")
    ratings = pd.read_csv("ratings.csv")
    similar_users = ratings[(ratings["movieId"] == movie_id)
                            & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users))
                        & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    all_users = ratings[(ratings["movieId"].isin(
        similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(10).merge(
        movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]


movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=True
)
recommendation_list = widgets.Output()


movie_name_input.observe(on_type, names='value')


#filme_input = input("Digite o nome de um filme para visualizar outros similares: ")

#lista_filmes_similares = find_similar_movies(5)
#print(lista_filmes_similares['title'])
#print(search("Tropa de elite"))
