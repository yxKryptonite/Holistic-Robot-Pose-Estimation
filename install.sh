conda create -n robopose python=3.9
conda activate robopose

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 # this is my pytorch version, please install pytorch based on your cuda version

# install pytorch3d, this may take a while
# please contact Yuxuan if you encounter any issue here
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

pip install -r requirements.txt