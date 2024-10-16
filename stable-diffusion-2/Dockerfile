FROM pytorch/torchserve-kfs:0.12.0-gpu

USER root

RUN mkdir /.cache && \
    chgrp -R 0 /home/model-server /.cache && \
    chmod -R g=u /home/model-server /.cache

USER model-server
