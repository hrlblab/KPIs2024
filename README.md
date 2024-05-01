# KPIs2024 example docker
KPIs challenge 2024

#Get our docker image for Training (Task 1)


        docker pull hrlblab333/kpis:1.0
    
Run the docker


        # you need to specify the input directory
        export input_dir=your_input_directory
        # you need to specify the output directory
        export output_dir=your_output_directory
        # make that directory
        mkdir $input_dir
        mkdir $output_dir
        #run the docker
        docker run --rm -v $input_dir:/input/:ro -v $output_dir:/output --gpus all -it hrlblab333/kpis:1.0

#Get our docker image for Validation (Task 1)

        docker pull hrlblab333/kpis:validation_patch   

        docker run --rm -v /Data/KPIs/data_train:/input/:ro -v /Data/KPIs/checkpoint:/model/:ro -v /Data/KPIs/validation_patch:/output --gpus all -it hrlblab333/kpis:validation_patch
        

#Get our docker image for Validation (Task 2)

        docker pull hrlblab333/kpis:validation_slide

        docker run --rm -v /Data/KPIs/data_val_slide:/input_slide/:ro -v /Data/KPIs/checkpoint:/model/:ro -v /Data/KPIs/validation_slide_20X:/output_slide -v /Data/KPIs/data_val_patch_20X:/input_patch -v /Data/KPIs/validation_slide_20X_patchoutput:/output_patch --gpus all -it hrlblab333/kpis:validation_slide

        
