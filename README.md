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
## The segmentation output should be at 20x magnification.
        
