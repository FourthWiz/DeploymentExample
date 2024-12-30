release: mlflow run . --no-conda
test: pytest
web: gunicorn src.serving.run:app -w 1 -k uvicorn.workers.UvicornWorker