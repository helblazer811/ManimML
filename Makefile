setup:
	conda activate manim; \
	export PROJECT_ROOT=$(pwd)
checkstyle:
	black .; \
	pydocstyle .
publish_pip:
	python3 -m build; \
	python3 -m twine upload --repository pypi dist/*