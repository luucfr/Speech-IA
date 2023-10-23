FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

WORKDIR /home

COPY ./train.py /home/train.py

RUN apt update
RUN apt install -y python3-pip
RUN pip3 install huggingsound
#RUN pip install --force-reinstall torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip3 install --force-reinstall torch torchvision torchaudio
RUN pip3 install accelerate -U

ARG NCCL_P2P_DISABLE=1
ARG CUDA_VISIBLE_DEVICES=0
ARG PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

CMD [ "torchrun", "train.py" ]
# CMD [ "torchrun", "--nproc_per_node", "1", "train.py" ]
