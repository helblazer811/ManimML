video:
	manim -pqh src/vae.py VAEScene --media_dir media
	cp media/videos/vae/1080p60/VAEScene.mp4 final_videos
train:
	cd src/autoencoder_models
	python vanilla_autoencoder.py
	python variational_autoencoder.py
checkstyle:
	pycodestyle src
	pydocstyle src