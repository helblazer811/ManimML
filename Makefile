video:
	manim -pqh src/autoencoder.py Autoencoder
checkstyle:
	pycodestyle src
	pydocstyle src
#docs:	
