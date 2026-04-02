import os
import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie
from openai import OpenAI
from dotenv import load_dotenv

class Command(BaseCommand):
    help = 'Calcula la similitud de coseno entre películas y un prompt usando OpenAI Embeddings'

    def handle(self, *args, **kwargs):
        # ✅ Cargar variables de entorno
        load_dotenv('openAI.env')
        
        client = OpenAI(api_key=os.environ.get('openai_apikey'))

        # ✅ 1. Seleccionar dos películas existentes
        try:
            movie1 = Movie.objects.get(title="Frankenstein")
            movie2 = Movie.objects.get(title="Cinderella")
        except Movie.DoesNotExist as e:
            self.stdout.write(self.style.ERROR(f"Error: No se encontró la película. Verifica los títulos en tu base de datos. {e}"))
            return

        # ✅ Función para obtener el embedding
        def get_embedding(text):
            response = client.embeddings.create(
                input=[text], 
                model="text-embedding-3-small"
            )
            return np.array(response.data[0].embedding, dtype=np.float32)

        # ✅ Función para calcular similitud de coseno
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        self.stdout.write(self.style.HTTP_INFO("Generando embeddings..."))

        # ✅ 2. Generar embeddings de las descripciones
        emb1 = get_embedding(movie1.description)
        emb2 = get_embedding(movie2.description)

        # ✅ 3. Calcular similitud entre las dos películas
        similarity = cosine_similarity(emb1, emb2)
        self.stdout.write(f"🎬 {movie1.title} vs {movie2.title}: {similarity:.4f}")

        # ✅ 4. Comparación contra un prompt (Cámbialo para tu actividad)
        prompt = "película de terror sobre un monstruo y un científico loco"
        prompt_emb = get_embedding(prompt)

        sim_prompt_movie1 = cosine_similarity(prompt_emb, emb1)
        sim_prompt_movie2 = cosine_similarity(prompt_emb, emb2)

        self.stdout.write(f"📝 Similitud prompt vs '{movie1.title}': {sim_prompt_movie1:.4f}")
        self.stdout.write(f"📝 Similitud prompt vs '{movie2.title}': {sim_prompt_movie2:.4f}")
        self.stdout.write("\n" + "="*50)