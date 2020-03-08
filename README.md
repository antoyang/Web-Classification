# Web-Classification
Our Kaggle submission on "French web domain classification" for the ALTEGRAD2020 MVA course (https://www.kaggle.com/c/fr-domain-classification), with Arnaud Massenet and Florent Rambaud (BERTExpress Team).

# Best Submission Summary
The text embeddings of our best submission are extracted using CamemBERT and TF-IDF+PCA and can be found at code/text_pipeline/Save (camembert_train_embeddings.pkl, camembert_test_embeddings.pkl, tfidf_emb_train.csv and tfidf_emb_test.csv).
They were generated using the codes code/text_pipeline/tfidf_embeddings.py and code/text_pipeline/CamemBERT.ipynb.
The classifier used is Linear Regression (with Grid Search and k-fold cross validation) and can be found at code/text_pipeline/classifier_lr.py.
The predictions were merged via test time augmentation, code available at code/text_pipeline/Test Time Augmentation/result_file_merge.ipynb.

# Other functionalities
We tried different approaches that are explained in Report.pdf.
