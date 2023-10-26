# Evolution of language diversity

This project investigates the connection of word polysemy evolution and perceived word concreteness.

The project is currently in development.
You can download and process the datasets for the German language by cloning the repository and running

```
python main.py -l german
```

This downloads word2vec embeddings from historic and contemporary sources, as well as concreteness ratings.
The processing includes deriving polysemy scores from the embeddings and calculating their change over time.

To recreate the results, you can run the `run.ipynb` notebook.
