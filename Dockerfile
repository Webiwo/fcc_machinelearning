FROM continuumio/anaconda3
WORKDIR /usr/src/app
# First, we copy only requirements.txt to use Docker's cache
COPY ../requirements.txt .
# Dependency installation (excluding pip cache - pip does not cache packages)
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir jupyterlab
COPY . .
WORKDIR /usr/src/app/nlp
# Flask port
EXPOSE 5000
# JupyterLab port
EXPOSE 8888

CMD ["python", "app.py"]


#docker build -t flask-nlp-app .
#docker run -p 5000:5000 -d flask-nlp-app
#docker run -p 8888:8888 -d flask-nlp-app jupyter lab --ip=0.0.0.0 --allow-root --no-browser