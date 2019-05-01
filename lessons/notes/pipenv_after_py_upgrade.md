After upgrading python, to resync pipenv run the following:
```
rm -rf `pipenv --venv`
pipenv install --dev
```