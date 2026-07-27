# Dockerfile for an open-source PyTorch project with CUDA support
# Base image: PyTorch 2.4.0 with CUDA 11.8 and cuDNN 9 runtime
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

# Set locale to support Chinese (zh_CN.UTF-8)
ENV LANG=zh_CN.UTF-8 \
    LANGUAGE=zh_CN:zh \
    LC_ALL=zh_CN.UTF-8

# Use a configurable APT mirror (default: Aliyun)
ARG APT_MIRROR=http://mirrors.aliyun.com/ubuntu
RUN sed -i "s|http://archive.ubuntu.com/ubuntu|${APT_MIRROR}|g" /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        vim locales sudo git openssh-client tmux && \
    locale-gen zh_CN.UTF-8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies for the project
RUN pip install --no-cache-dir gym scipy matplotlib pytest plotly seaborn wandb

# Define a configurable username (default: zhangheng)
ARG USERNAME=zhangheng
# Create a non-root user with sudo privileges
RUN groupadd -r ${USERNAME} && useradd -m -r -g ${USERNAME} ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set up project directory with correct ownership
RUN mkdir -p /home/${USERNAME}/last-dance && chown ${USERNAME}:${USERNAME} /home/${USERNAME}/last-dance

# Switch to non-root user
USER ${USERNAME}
WORKDIR /home/${USERNAME}

# Copy project files into the container
COPY --chown=${USERNAME}:${USERNAME} . /home/${USERNAME}/last-dance/

# Expose port 8888 (e.g., for Jupyter Notebook)
EXPOSE 8888

# Default command: start an interactive bash shell
CMD ["bash"]