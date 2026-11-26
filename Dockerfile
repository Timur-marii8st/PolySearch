FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY polysearch ./polysearch
RUN pip install --no-cache-dir ".[server]" && useradd --uid 10001 --create-home appuser
USER 10001:10001
EXPOSE 8000
CMD ["python","-m","uvicorn","polysearch.api:create_app","--factory","--host","0.0.0.0","--port","8000","--no-proxy-headers"]
