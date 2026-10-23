ARG RUNTIME=python:3.12-slim

FROM --platform=$BUILDPLATFORM python:3.12-slim AS framework-wheel

RUN pip install --no-cache-dir --disable-pip-version-check poetry==2.5.1

WORKDIR /opt/framework

COPY --from=framework pyproject.toml LICENSE ./
COPY --from=framework src/eemls ./src/eemls
RUN poetry build --format wheel --output /wheels

FROM ${RUNTIME} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN python -m venv /opt/venv

WORKDIR /opt/eemls

COPY pyproject.toml poetry.lock README.md ./
RUN python -m venv /opt/poetry \
    && /opt/poetry/bin/pip install poetry==2.5.1 poetry-plugin-export==1.10.1 \
    && /opt/poetry/bin/poetry export --only main --output locked.txt \
    && awk '/^[^ ]/ { framework = /^eemls @/ } !framework' locked.txt > requirements.txt \
    && pip install --no-deps --require-hashes --requirement requirements.txt

COPY framework.txt ./
RUN --mount=type=bind,from=framework-wheel,source=/wheels,target=/opt/framework \
    pip install --no-deps --no-index --require-hashes --requirement framework.txt --find-links /opt/framework

COPY src ./src
RUN pip install --no-deps .

FROM ${RUNTIME}

RUN useradd --system --uid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin eemls

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    EEMLS_CONFIG_DIR=/opt/eemls/config

WORKDIR /opt/eemls

COPY --from=builder /opt/venv /opt/venv

USER eemls

EXPOSE 8083
ENTRYPOINT ["eemls-observation-service"]
