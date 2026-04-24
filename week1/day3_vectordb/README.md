Base sentence: 'I love machine learning'
Comparison                                      Score  Interpretation
---------------------------------------------------------------------------
  I enjoy ML a lot                             0.4432  loosely related
  AI and deep learning are fascinating         0.6002  related topic
  The weather is nice today                    0.0701  unrelated
  I hate machine learning                      0.8005  very similar
  I love machine learning                      1.0000  unknown


============================================================
EXPERIMENT 1: Semantic search vs keyword search
============================================================

Query: 'cats and other domestic animals'
(Note: query contains 'cats' — doc1 has 'cat', doc3 has no matching keyword)

  Rank 1: [0.4890] Felines are known for being independent.
  Rank 2: [0.2994] The cat sat on the mat.
  Rank 3: [0.1364] A dog is running in the park.
  Rank 4: [0.0973] Neural networks mimic the human brain.
  Rank 5: [0.0447] Python is used for data science.

KEY INSIGHT: If rank 1 is doc3 (Felines...) not doc1 (cat on mat),
it means search is by MEANING, not keyword. 'Felines' = 'cats' semantically.


============================================================

Ingested 'python_guide': 2 chunks
Ingested 'hr_policy': 2 chunks
Total chunks: 4

============================================================

Q: How many leave days do employees get?
Filter: category='hr'
A: Employees are entitled to 20 days of paid leave per year.
----------------------------------------


add() vs upsert()

add() fucntion need new id if id already exist it will fail, but upsert() fucntion update document if id already exist , intesrt new id if not exist.  