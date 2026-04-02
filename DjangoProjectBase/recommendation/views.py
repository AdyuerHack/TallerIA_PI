from django.shortcuts import render
from movie.models import Movie
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import os


def get_embedding(text):
    load_dotenv('openAI.env')
    client = OpenAI(api_key=os.environ.get('openai_apikey'))
    response = client.embeddings.create(input=[text], model="text-embedding-3-small")
    return np.array(response.data[0].embedding, dtype=np.float32)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def recommendation(request):
    searchTerm = request.GET.get('searchMovie')
    best_movie = None
    max_similarity = -1

    if searchTerm:
        try:
            prompt_emb = get_embedding(searchTerm)

            for movie in Movie.objects.all():
                if movie.emb:
                    movie_emb = np.frombuffer(movie.emb, dtype=np.float32)
                    similarity = cosine_similarity(prompt_emb, movie_emb)

                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_movie = movie
        except Exception as e:
            print(f"Error: {e}")

    return render(request, 'recommendation.html', {
        'searchTerm': searchTerm,
        'best_movie': best_movie,
        'similarity': f"{max_similarity:.4f}" if max_similarity > 0 else None,
    })
