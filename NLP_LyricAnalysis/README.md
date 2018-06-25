# NLP Lyric Analysis  
There have been several approaches to analyzing music based off audio features, trying to extract different components such as artist, sentiment, or genre.  Lyrics are a component of the audio but generally are not utilized solely to make inferences about a particular song.  In this project, utilizing ideas from natural language processing (NLP) we used the supervised learning classifiers Naive Bayes and Support Vector Machine to predict song sentiment.  Furthermore, sentence structure was ignored and only the frequency of words was used to curate the sentiment of songs.  The results were above random but were only about 41% accurate.  This may have been a result of poor ground truth data but did show promise that lyrics alone can be a powerful tool in classifying the sentiment behind songs.
## HOW TO RUN:
1. $python3 server.py -p 9001

2. In browser type localhost:9001

3. Enter artist name and song name then press Predict Song

### NLP:
*tokenizing
*removal of punctuation
*Stop words
*Porter Stemming algorithm
*tf-idf vectors
*normalizing the vectors

### Clustering:
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
