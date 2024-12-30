test: mlflow run . --no-conda && pytest
web: gunicorn src.serving.run:app -w 1 -k uvicorn.workers.UvicornWorker