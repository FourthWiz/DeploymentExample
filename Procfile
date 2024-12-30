release: mlflow run .
test: pytest
web: gunicorn src.serving.run:app -k uvicorn.workers.UvicornWorker