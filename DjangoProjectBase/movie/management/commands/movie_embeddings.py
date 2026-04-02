from django.core.management.base import BaseCommand
from movie.models import Movie
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np

class Command(BaseCommand):
    help = 'Generate and store OpenAI embeddings for all movies in the database'

    def handle(self, *args, **kwargs):
        load_dotenv('openAI.env')
        client = OpenAI(api_key=os.environ.get('openai_apikey'))

        movies = Movie.objects.all()
        self.stdout.write(f"Found {movies.count()} movies in the database")

        for movie in movies:
            try:
                # Generate embedding for the movie description
                descricao = movie.description if movie.description else movie.title
                response = client.embeddings.create(
                    input=[descricao],
                    model="text-embedding-3-small"
                )
                
                # Convert embedding to numpy array and then to bytes
                embedding_array = np.array(response.data[0].embedding, dtype=np.float32)
                movie.emb = embedding_array.tobytes()
                movie.save()
                
                self.stdout.write(self.style.SUCCESS(f"👌 Embedding stored for: {movie.title}"))
            except Exception as e:
                self.stderr.write(f"Error generating embedding for {movie.title}: {e}")

        self.stdout.write(self.style.SUCCESS("🌟 Finished generating embeddings for all movies"))
