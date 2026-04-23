=================================================================
SIMILARITY INTUITION TEST
=================================================================
Pair                                                     Score
-----------------------------------------------------------------
"I love cats..." vs "I adore kittens..."                 0.732
"I love cats..." vs "Cats are my favorite..."            0.845
"I love cats..." vs "The stock market cra..."            0.056
"Machine learning is ..." vs "ML is difficult..."        0.559
"Python is a programm..." vs "Python is a snake..."      0.680

Notice the last pair — 'Python the language' vs 'Python the snake'.
This is a known limitation of sentence embeddings. Keep it in mind.

=================================================================
RAG pipepile have 5 steps 
=================================================================
step 1 - we chunk the document.
step 2 - store it in vector DB.
step 3 - find the best sutable chunks for user query.
step 4 - build prompt using context and user query 
step 5 - provide the sutable context to llm and ask to use the provided context only for response

if user ask question which is not present in context ot document it will response "I don't have enough information to answer this."
 