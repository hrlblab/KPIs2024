## Pull from existing image
FROM nvcr.io/nvidia/pytorch:21.05-py3
  
## Copy requirements
COPY ./requirements.txt .

## Install Python packages in Docker image

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install tifffile


## Copy all files (here "./src/run_inference.py")
COPY ./ ./


RUN mkdir /myhome/
COPY ./src /myhome
RUN chmod -R 777 /myhome


## Execute the inference command 
CMD ["./src/unet_validation_slide.py"]
ENTRYPOINT ["python3"]
