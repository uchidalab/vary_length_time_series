FROM tensorflow/tensorflow:2.4.1-gpu-jupyter 
LABEL maintainer="Brian Kenji Iwana"

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && apt-get install -y python3-tk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install simpletransformers
RUN pip install wandb
RUN pip install git+https://github.com/albermax/innvestigate@1.0.8.3
RUN pip install scikit-image==0.15.0
RUN pip install tqdm
RUN pip install pydot
RUN pip install graphviz
RUN pip install transforms3d
RUN pip install seaborn
RUN pip install nltk
RUN pip install -U imbalanced-learn
