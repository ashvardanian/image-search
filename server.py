from ucall.rich_posix import Server
import uform
import usearch

import numpy as np
from PIL import Image


server = Server()
model = uform.get_model('unum-cloud/uform-vl-multilingual')
index = usearch.Index(dim=256)


@server
def add(label: int, photo: Image.Image):
    image = model.preprocess_image(photo)
    vector = model.encode_image(image).detach().numpy()
    labels = np.array([label], dtype=np.longlong)
    index.add(labels, vector, copy=True)


@server
def search(query: str) -> np.ndarray:
    tokens = model.preprocess_text(query)
    vector = model.encode_text(tokens).detach().numpy()
    neighbors = index.search(vector, 3)
    return neighbors[0][:neighbors[2][0]]


server.run()
