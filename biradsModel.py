import pickle
import os
from django.conf import settings

path=os.path.join(settings.MEDIA_ROOT, 'models', 'modelo_naive_bayes.pkl')
with open(path, 'rb') as archivo:
    naiveModel = pickle.load(archivo)
def predict_naive(input):
    return naiveModel.predict_proba(input)