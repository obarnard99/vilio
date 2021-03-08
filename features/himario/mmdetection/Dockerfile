ARG PYTORCH="1.5"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 htop\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==latest+torch1.5.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
RUN echo ""
RUN pip install mmcv==1.1.1 loguru fire

# Install MMDetection
RUN conda clean --all
RUN pwd
# RUN git clone https://github.com/HimariO/mmdetection.git /mmdetection
COPY . /mmdetection
WORKDIR /mmdetection

ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
RUN apt-get update && apt-get install 'ffmpeg' 'libsm6' 'libxext6' vim -y
RUN sed -i "1iforce_color_prompt=true" ~/.bashrc

# RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
#     && apt-get install -y apt-transport-https ca-certificates curl \
#     && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
#     && apt-get update \
#     && apt-get install -y google-cloud-sdk