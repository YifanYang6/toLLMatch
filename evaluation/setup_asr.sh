cd ../whisper-jax && pip install -e . && cd ../evaluation
pip install -U jax[cuda12]==0.4.28
pip install -U jaxlib==0.4.28+cuda12.cudnn89  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax==0.8.4 # fix jax version confict of `orbax-checkpoint` required by flax
pip install -U transformers==4.34 # fix whisper download index issue https://github.com/huggingface/transformers/issues/28156
pip install accelerate==0.31.0 # for using device mapping
pip install termcolor==2.4.0
pip install python-dotenv==1.0.1
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
pip install numpy==1.23.5 # fix jax numpy version conflict
sed -i '38s/^/#/' /app/toLLMatch/evaluation/.venv/lib/python3.10/site-packages/jaxlib/gpu_triton.py # fix jaxlib python binding