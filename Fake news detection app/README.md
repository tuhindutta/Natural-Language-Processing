1. Run main.py to launch the app.
2. ./Fake_News/ contains the data, model generator code and the pickled model
3. Check ./Fake_News/fake-news-classifier-accuracy-99.ipynb to understand the model generation steps.


After launching the app, write the news headline in the space provided, click 'Classify' to generate result in the terminal window.



Model Generator :- After proper hyperparameter tuning, Bernoulli Naive Bayes algorithm has been choosen. Due to lack of computational resources, training is performed on only 1000 true and fake news data. On availability of computational resources, training could be performed on the complete dataset. Moreover, LSTM or any neural network algorithm can be used to develop the model and its working.
