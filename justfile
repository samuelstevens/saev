lint: fmt
    ruff check saev/ main.py analysis.py generate_app_data.py

fmt:
    isort .
    ruff format --preview .
