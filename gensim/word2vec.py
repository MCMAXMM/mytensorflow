# documents = ["Human machine interface for lab abc computer applications",
#              "A survey of user opinion of computer system response time",
#              "The EPS user interface management system",
#              "System and human system engineering testing of EPS",
#              "Relation of user perceived response time to error measurement",
#              "The generation of random binary unordered trees",
#              "The intersection graph of paths in trees",
#              "Graph minors IV Widths of trees and well quasi ordering",
#              "Graph minors A survey"]
# stoplist = set('for a of the and to in'.split())
# texts = [[word for word in document.lower().split() if word not in stoplist]
#          for document in documents]
#
# # remove words that appear only once
# from collections import defaultdict
# frequency = defaultdict(int)
# for text in texts:
#     for token in text:
#         frequency[token] += 1
#
# texts = [[token for token in text if frequency[token] > 1] for text in texts]
#
# from pprint import pprint  # pretty-printer
# pprint(texts)
# import gensim
# dictionary = corpora.Dictionary(texts)
# dictionary.save(os.path.join(TEMP_FOLDER, 'deerwester.dict'))  # store the dictionary, for future reference
# print(dictionary)
# print(dictionary.token2id)
# new_doc = "Human computer interaction"
# new_vec = dictionary.doc2bow(new_doc.lower().split())
# print(new_vec)#其中 bagofwords 采用的稀疏向量的表达形式
#
# corpus = [dictionary.doc2bow(text) for text in texts]
# corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'deerwester.mm'), corpus)  # store to disk, for later use
# for c in corpus:
#     print(c)




# import logging
#
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# import tempfile
# import os.path
#
# TEMP_FOLDER = tempfile.gettempdir()
# print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))
# from gensim import corpora, models, similarities
# if os.path.isfile(os.path.join(TEMP_FOLDER, 'deerwester.dict')):
#     dictionary = corpora.Dictionary.load(os.path.join(TEMP_FOLDER, 'deerwester.dict'))
#     corpus = corpora.MmCorpus(os.path.join(TEMP_FOLDER, 'deerwester.mm'))
#     print("Used files generated from first tutorial")
# else:
#     print("Please run first tutorial to generate data set")
# print(dictionary[0])
# print(dictionary[1])
# print(dictionary[2])
# print(dictionary)
# tfidf = models.TfidfModel(corpus)
#
# doc_bow = [(0, 1), (1, 1)]
# print(tfidf[doc_bow])
# corpus_tfidf = tfidf[corpus]
# for doc in corpus_tfidf:
#     print(doc)
# lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize an LSI transformation
# corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
# lsi.print_topics(2)
# for doc in corpus_lsi: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
#     print(doc)
# lsi.save(os.path.join(TEMP_FOLDER, 'model.lsi'))  # same for tfidf, lda, ...
# # lsi = models.LsiModel.load(os.path.join(TEMP_FOLDER, 'model.lsi'))


from gensim.models import word2vec
import logging
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",level=logging.INFO)
# raw_sentences=["the quick brown fox jumps over the lazy dogs","yoyoyo you go home now to sleep"]
# sentences=[s.split() for s in raw_sentences]
# print(sentences)
# model=word2vec.Word2Vec(sentences,min_count=1,size=200)
# s=model.similarity("dogs","you")
# model.save("word2vec.model")
# print(model["dogs"])
model=word2vec.Word2Vec.load("word2vec.model")
print(model["dogs"])
