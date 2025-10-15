import os, shutil
os.makedirs('data/raw', exist_ok=True)
# Copy the included sample to titanic.csv so the project runs immediately.
if not os.path.exists('data/raw/titanic.csv'):
    shutil.copy('data/raw/titanic_sample.csv', 'data/raw/titanic.csv')
    print('Copied sample dataset to data/raw/titanic.csv')
else:
    print('data/raw/titanic.csv already exists')
