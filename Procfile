build: mlflow run .
test: pytest
web: gunicorn src.serving.run:app -w 4 -k uvicorn.workers.UvicornWorker