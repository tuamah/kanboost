FROM python:3.11-slim

WORKDIR /app

# kanboost itself pulls in torch/pykan (heavy); [api] adds fastapi/uvicorn
RUN pip install --no-cache-dir "kanboost[api]"

# Mount or COPY a trained model into /app/model.pt before running, e.g.:
#   docker run -v /path/to/model.pt:/app/model.pt ...
ENV KANBOOST_MODEL_PATH=/app/model.pt
ENV KANBOOST_DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "kanboost.serving:app", "--host", "0.0.0.0", "--port", "8000"]
