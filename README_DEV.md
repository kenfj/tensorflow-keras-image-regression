# Development Note

Initial setup pipenv commands to start development

```bash
brew update
brew install pyenv pipenv python # Python 3

export PIPENV_VENV_IN_PROJECT=true
pipenv --python 3.8

# https://code.visualstudio.com/docs/python/linting
pipenv install --dev autopep8 pylint
pipenv install tensorflow==2.5.0
pipenv install pandas matplotlib scikit-learn seaborn

# https://yamap55.hatenablog.com/entry/2018/07/22/235746
pylint --generate-rcfile > .pylintrc

# https://stackoverflow.com/questions/33961756
# generated-members=numpy.*,np.*,pandas.*,pd.*

# disable=expression-not-assigned,pointless-statement
```

pipenv misc commands

```bash
# list packages
pipenv run pip list

# upgrade packages
pipenv update --outdated
pipenv update

# remove unused packages
pipenv clean

# remove .venv to restart setup
pipenv --rm
```
