## FastMRI Project(Deep Learning OMSCS - Spring 2021)



### Dependencies

The code was tested on the following machines:

1. Debian 10 (Google cloud)
2. Ubuntu 18.04(vast.ai)

Using:
1. Cuda 11.0 which was preinstalled
2. Python3.8 which was installed using this [link](https://tecnstuff.net/how-to-install-python-3-8-on-debian-10/)


### Project Structure

The top level directory of our code is named CS7643. It contains the Project directory and this README.md file.

Under Project are several important files and directories. We mention some relevant ones here.

experiments: This is where the run script for each model resides under a sub-directory named for the model. This is also where reconstructed images are generated.

data: This is where the data downloaded from the fastMRI project is placed and unzipped. Once unzipped, it creates 3 directories - singlecoil_train, singlecoil_val and singlecoil_test (rename this one to remove the v2)

results: This has sub-directories for each model under which we have checkpoints files for best epochs.

graphs: This module has the .ipynb file which is used to generate the validation and training loss for each model.

fastmri/pl_modules: This is where the PyTorch Lightning module for each model resides

fastmri/models: This is where the code for the models resides

fastmri/reconstruct.py: Code for image reconstruction.

setup.py, requirements.txt: For installing tool dependencies as explained below


### Installations

1. Set the base directory variable for the project in the environment:<br />
export PROJECT_BASE=&lt;directory where project would be checked out&gt;
2. Check project code out from GitHub. You will see the CS7643 directory upon successful checkout.
3. Follow the steps below using the shell commandline to install required Python packages.<br />
cd $PROJECT_BASE/CS7643/Project<br />
pip3.8 install -e setup.py<br />
pip3.8 install -e dev-requirements.txt <br />

### Dataset

We requested the fastMRI from the [NYU fastMRI](https://fastmri.med.nyu.edu) website and Once approval is received, follow directions in the email to download single-coil knee dataset using curl to store the data in the google cloud bucket(using gsutils) or store them directly on the server. We have used the singlecoil - train, validation and test data set.

On the environment<br />
cd $PROJECT_BASE/CS7643/Project/data<br />
curl -C &lt;URL&gt; --output knee_singlecoil_train.tar.gz <br />
curl -C &lt;URL&gt; --output knee_singlecoil_val.tar.gz <br />
curl -C &lt;URL&gt; --output knee_singlecoil_test_v2.tar.gz <br />

tar -zxvf knee_singlecoil_train.tar.gz<br />
tar -zxvf knee_singlecoil_val.tar.gz<br />
tar -zxvf knee_singlecoil_test_v2.tar.gz<br />

mv knee_singlecoil_test_v2 knee_singlecoil_test

There should be now three subdirectories under $PROJECT_BASE/CS7643/Project/data with .h5 data files<br />
knee_singlecoil_train<br />
knee_singlecoil_val<br />
knee_singlecoil_test<br />

### Running the Models

On the environment, navigate to the experiments directory:<br />
cd $PROJECT_BASE/CS7643/Project/experiments<br />
cd &lt;model_name&gt;<br />

A training experiment is run using the general command below: <br />
nohup python3.8 run_&lt;model_name&gt;.py --mode=train --max_epochs=&lt;max_epochs (default 50)&gt; --sample_rate=1.0 &gt; nohup.out 2&gt;&1 &

Here are the specific commands for each model. The training run log is stored in nohup.out and can be seen using the command<br />
tail -f nohup.log

| Model Name  | Command to run                                               | Log file  |
| ----------- | ------------------------------------------------------------ | --------- |
| Unet        | nohup python3.8 CS7643/Project/experiments/unet/run_unet.py --mode=train --max_epochs=20 --sample_rate=1.0 >>nohup.out 2>&1 & | nohup.out |
| R2Unet      | nohup python3.8 CS7643/Project/experiments/r2unet/run_r2unet.py --mode=train --max_epochs=20 --sample_rate=1.0 >>nohup.out 2>&1 & | nohup.out |
| Attn Unet   | nohup python3.8 CS7643/Project/experiments/att_unet/run_att_unet.py --mode=train --max_epochs=20 --sample_rate=1.0 >>nohup.out 2>&1 & | nohup.out |
| Dense Unet  | nohup python3.8 CS7643/Project/experiments/denseunet/run_dense_unet.py --mode=train --max_epochs=20 --sample_rate=1.0 >>nohup.out 2>&1 & | nohup.out |
| Nested Unet | nohup python3.8 CS7643/Project/experiments/nested_unet/run_nested_unet.py --mode=train --max_epochs=20 --sample_rate=1.0 >>nohup.out 2>&1 & | nohup.out |
| Resnet      | nohup python3.8 CS7643/Project/experiments/resnet/run_resnet.py --mode=train --max_epochs=20 --sample_rate=1.0 >>nohup.out 2>&1 & | nohup.out |


### Plotting the Training results

After this we used the generate_graph.ipynb file ( under graphs folder) which was used to generate the validation and training loss. The folder for different output files generated from the above table were kept along with folder name in the same directory as ipynb file. All the cells are run in order in the .ipynb and the final images are generated for validation and training loss.

### 
### Testing the Models
Testing is done using the same scripts in the same directories as training but running them in the test mode. Here is the general command.<br />

python3.8 run_&lt;model_name&gt;.py --mode=test --max_epochs=1 --sample_rate=1.0

Here are the specific commands for each model. Running the test also generates the reconstructed images under the $PROJECT_BASE/CS7643/Project/experiments/&lt;model_name&gt;/&lt;testfile_name&gt; directory. We generate images for all files by picking slice 22 for reconstruction since it shows a complete knee.

| Model Name  | Command to run                                               |
| ----------- | ------------------------------------------------------------ |
| Unet        | python3.8 CS7643/Project/experiments/unet/run_unet.py --mode=test |
| R2unet      | python3.8 CS7643/Project/experiments/r2unet/run_r2unet.py --mode=test |
| Attn Unet   | python3.8 CS7643/Project/experiments/att_unet/run_att_unet.py --mode=test |
| Dense Unet  | python3.8 CS7643/Project/experiments/denseunet/run_dense_unet.py --mode=test |
| Nested Unet | python3.8 CS7643/Project/experiments/nested_unet/run_nested_unet.py --mode=test |
| Resnet      | python3.8 CS7643/Project/experiments/resnet/run_resnet.py --mode=test |

### Code References

The major code is referenced from https://github.com/facebookresearch/fastMRI but this one only contains unet. For other models which were used, we used [Papers with code](https://paperswithcode.com/) and tried to use the models in conjunction with the facebook research code.

