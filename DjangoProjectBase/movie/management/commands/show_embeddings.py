from django.core.management.base import BaseCommand
from movie.models import Movie
import numpy as np

class Command(BaseCommand):
    help = 'Show the first 5 elements of a random movie embedding'

    def handle(self, *args, **kwargs):
        # We will get a random movie or just pick the first one
        movie = Movie.objects.order_by("?").first()
        
        if not movie:
            self.stderr.write("No movies found in database.")
            return

        try:
            embedding_vector = np.frombuffer(movie.emb, dtype=np.float32)
            self.stdout.write(self.style.SUCCESS(f"Movie: {movie.title}"))
            self.stdout.write(f"First 5 embedding values: {embedding_vector[:5]}")
        except Exception as e:
            self.stderr.write(f"Error reading embedding for {movie.title}: {e}")
