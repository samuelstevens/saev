lint: fmt
    ruff check saev/ main.py analysis.py generate_app_data.py

fmt:
    fd -e py | xargs isort
    fd -e py | xargs ruff format --preview
