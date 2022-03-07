setup:
	conda activate manim
	export PROJECT_ROOT=$(pwd)
video:
	manim -pqh src/vae.py VAEScene --media_dir media
	cp media/videos/vae/720p60/VAEScene.mp4 examples
train:
	cd src/autoencoder_models
	python vanilla_autoencoder.py
	python variational_autoencoder.py
generate_visualizations:
	cd src/autoencoder_models
	python generate_images.py
	python generate_interpolation.py
	python generate_disentanglement.py
checkstyle:
	pycodestyle src
	pydocstyle src