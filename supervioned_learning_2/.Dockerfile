FROM python:3.11.5

LABEL DESCRIPTION = "This is a machine learning model image"
LABEL MAINTAINER = "pviniciuspsilva@gmail.com"
LABEL VERSION = 1.0

ENV APP_HOME /app_home
WORKDIR ${APP_HOME}

COPY ./data ${APP_HOME}/
COPY ./src ${APP_HOME}
COPY ./requirements.txt ${APP_HOME}
COPY ./setup.py ${APP_HOME}
COPY ./docs ${APP_HOME}
COPY ./entrypoint.sh ${APP_HOME}

RUN python -m pip install --no-cache-dir- -r requirements.txt
RUN chmod +x

ENTRYPOINT ["./entrypoint.sh"]