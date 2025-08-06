# query_rewriter.py
from InstructorEmbedding import INSTRUCTOR

# Load the Instructor model once
instructor_model = INSTRUCTOR('hkunlp/instructor-xl')

def generate_semantic_query(prompt: str, theme: str, main: str, sub: str) -> list:
    """
    Generate a semantically rich embedding for a user prompt using thematic context.
    """
    task_instruction = "Represent the user query for retrieving Jewish Torah sources:"
    combined_input = (
        f"Theme: {theme}\n"
        f"Main Category: {main}\n"
        f"Subcategory: {sub}\n"
        f"User Question: {prompt}"
    )
    embedding = instructor_model.encode([[task_instruction, combined_input]])[0]
    return embedding.tolist()

def get_instructor_model():
    """
    Return the shared instance of the instructor model for reuse across scripts.
    """
    return instructor_model
