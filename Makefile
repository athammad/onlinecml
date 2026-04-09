
set_env:
	pyenv virtualenv 3.12.9 ocml_env
	pyenv local ocml_env

install_reqs:
	python -m pip install --upgrade pip
	@pip install -r requirements.txt
