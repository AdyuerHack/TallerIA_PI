from django.core.management.base import BaseCommand
from movie.models import Movie
import os

class Command(BaseCommand):
    help = 'Actualiza las rutas de las imágenes de las películas desde la carpeta local media/movie/images/'

    def handle(self, *args, **kwargs):
        # 1. Obtenemos todas las películas de la base de datos
        movies = Movie.objects.all()
        self.stdout.write(f"Se encontraron {movies.count()} películas.")

        count = 0
        for movie in movies:
            # 2. Definimos el nombre del archivo basado en el título
            # El formato entregado es: m_NOMBRE_PELICULA.png
            image_filename = f"m_{movie.title}.png"
            
            # 3. Definimos la ruta relativa que Django guarda en el ImageField
            # Nota: Django concatena esto con MEDIA_URL automáticamente
            image_relative_path = os.path.join('movie/images', image_filename)

            # 4. Actualizamos el campo 'image' del modelo
            movie.image = image_relative_path
            movie.save()
            
            self.stdout.write(self.style.SUCCESS(f"Imagen vinculada para: {movie.title}"))
            count += 1

        self.stdout.write(self.style.SUCCESS(f"Proceso finalizado. {count} películas actualizadas."))