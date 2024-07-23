# KPIs2024
KPIs challenge 2024

# Training example docker
## Get our docker image for Training (Task 1)


        docker pull hrlblab333/kpis:1.0
    
## Run the docker


        # you need to specify the input directory
        export input_dir=your_input_directory
        # you need to specify the output directory
        export output_dir=your_output_directory
        # make that directory
        mkdir $input_dir
        mkdir $output_dir
        #run the docker
        docker run --rm -v $input_dir:/input/:ro -v $output_dir:/output --gpus all -it hrlblab333/kpis:1.0




# Validation example docker
## Get our docker image for Validation (Task 1)

        docker pull hrlblab333/kpis:validation_patch   

        # you need to specify the input, output directory, and the model path
        docker run --rm -v $input_dir:/input/:ro -v $model:/model/:ro -v $output_dir:/output --gpus all -it hrlblab333/kpis:validation_patch

## File structure (task1)
The directory for both training and validation needs to have the following structure:
        
```bash
input_dir
    └── 56NX
        └── case1
            └── img
                └── patch1.jpg
                └── patch2.jpg
                ...
            └── mask
                └── patch1_mask.jpg
                └── patch2_mask.jpg
                ...
        └── case2
        └── case3
        ...
    └── DN
    └── NEP25
    └── normal

```

## Get our docker image for Validation (Task 2)

        docker pull hrlblab333/kpis:validation_slide

        # you need to specify the input, output directory, and the model path
        # you can specify the patch_save, and patch_mask_save directories to save the middle product for further assessment.
        docker run --rm -v $input_dir:/input_slide/:ro -v $model:/model/:ro -v $output_dir:/output_slide -v $patch_save:/input_patch -v $patch_mask_save:/output_patch --gpus all -it hrlblab333/kpis:validation_slide
        
## File structure (task2)
The directory for validation, task2 need to have the following structure:

```bash
input_dir
    └── 56NX
        └── img
            └── case1.tiff
            └── case2.tiff
            ...
        └── mask
            └── case1_mask.tiff
            └── case2_mask.tiff
            ...
    └── DN
    └── NEP25
    └── normal

```
## The segmentation output should be at 20x magnification(Optical magnification).
        
# Docker preparation for challenge submission
We provide the source code of the Docker file for Task 2 in this repository. Below is a step-by-step tutorial on how to build a Docker container for your submission.

Follow this [link](https://docs.docker.com/engine/install/) to install Docker on your machine.

The structure of the Docker folder should be as follows:

```bash
docker-folder
    └── src
        └── unet_validation_slide.py
    └── Dockerfile       
    └── requirements.txt

```

`unet_validation_slide.py` is your inference code. 
`Dockerfile` contains scripts to install packages and run your code in Docker.
`requirements.txt` lists the virtual environment packages, which you can directly export from your local environment.

In your inference code, replace all absolute paths with reference paths that you will use in your Docker command line (`/input_slide/`,`/model/`,`/output_slide/`, etc.)

Follow the step-by-step commands in [CS-MIL_docker_commandline.txt](https://github.com/hrlblab/KPIs2024/blob/main/Validation_slide_docker/CS-MIL_docker_commandline.txt)

        # Create the docker container
        docker build -f Dockerfile -t kpis .
        
You may encounter errors while installing required packages or running your Docker container. Modify the scripts in `Dockerfile` to ensure all packages are installed and your inference code runs without errors.
        
        docker login
        docker tag kpis <your_user_name>/kpis:validation_slide
        docker push <your_user_name>/kpis:validation_slide
        
        # Run the code online with GPU
        docker pull <your_user_name>/kpis:validation_slide
        docker run --rm -v <your_local_folder_to_store_slides>:/input_slide/:ro -v <your_local_folder_to_store_model>:/model/:ro -v <your_local_folder_to_store_output_slides>:/output_slide -v <your_local_folder_to_store_cropped_patches>:/input_patch -v <your_local_folder_to_store_output_patches>:/output_patch --gpus all -it <your_user_name>/kpis:validation_slide
        
        # Run the Docker locally with GPU
        docker run --rm -v <your_local_folder_to_store_slides>:ro -v <your_local_folder_to_store_model>:ro -v <your_local_folder_to_store_output_slides>:/output_slide -v <your_local_folder_to_store_cropped_patches>:/input_patch -v <your_local_folder_to_store_output_patches>:/output_patch --gpus all -it kpis


You can also include your model weights in your `src` folder if you prefer not to share an additional download link for your model.

For more information, you may refer to another Docker tutorial for challenges [here](https://www.synapse.org/Synapse:syn51236108/wiki/622826).
