test: mlflow run . --env-manager=local && pytest
web: gunicorn src.serving.run:app -w 1 -k uvicorn.workers.UvicornWorker