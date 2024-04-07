# KPIs2024
KPIs challenge 2024

#Get our docker image


        sudo docker pull hrlblab333/kpis:1.0
    
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
