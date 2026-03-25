FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg
ENV DOCS_HOST=0.0.0.0
ENV DOCS_PORT=8000

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY docs ./docs
COPY data ./data

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .[docs]

EXPOSE 8000

CMD sh -c "python -m sphinx -b html -W --keep-going docs docs/_build/html && cd docs/_build/html && python -m http.server ${DOCS_PORT} --bind ${DOCS_HOST}"
