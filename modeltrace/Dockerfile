# ModelTrace analysis service. Standalone image: no GPT-Load code is copied in,
# so it can be rebuilt, updated, and rolled back on its own schedule.
FROM python:3.14-slim@sha256:0214bda9af2652a418f2bf34373a4d3ca00a5f4784081e040049433ac9cc79d5 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELTRACE_HOST=0.0.0.0 \
    MODELTRACE_PORT=8080

WORKDIR /app

COPY modeltrace/requirements.txt ./modeltrace/requirements.txt
RUN pip install --no-cache-dir -r modeltrace/requirements.txt \
    && addgroup --system --gid 10001 modeltrace \
    && adduser --system --uid 10001 --ingroup modeltrace --no-create-home modeltrace

COPY modeltrace ./modeltrace
COPY LICENSES/MIT.txt /app/licenses/MIT.txt
COPY THIRD_PARTY_NOTICES.md /app/licenses/THIRD_PARTY_NOTICES.md

USER 10001:10001
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4).status == 200 else 1)"]

ENTRYPOINT ["python", "-m", "modeltrace.server"]
