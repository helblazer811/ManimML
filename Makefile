video:
	manim -pqh src/vae.py VAEScene --media_dir media
	cp media/videos/vae/1080p60/VAEScene.mp4 final_videos
checkstyle:
	pycodestyle src
	pydocstyle src