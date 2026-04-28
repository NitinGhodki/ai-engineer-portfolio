============================================================
TOKEN COST COMPARISON SUMMARY
============================================================
Pattern                Prompt tokens Use case
------------------------------------------------------------
  zero_shot                      ~5-20  simple tasks, known domains
  few_shot                     ~50-200  format/tone control needed
  chain_of_thought     ~20-50 in, 200-600 out  multi-step reasoning
  structured_output            ~80-150  machine-readable output
  self_consistency             ~3x any  high-stakes, unreliable outputs
  injection_defense           ~0 extra  always — it is free



  Standard vs few-shot

  Question: What is the refund policy?
  Standard:  Our refund policy allows full refunds within 30 days of purchase. After 30 days, refunds are evaluated case by case.
  Few-shot:  Our refund policy allows full refunds within 30 days of purchase. After 30 days, refunds are evaluated case by case.


 when would you NOT use chain-of-thought in a production system?
 In simple task, when need fast response or when there is creative content generation, tihs conditionn we must not use COT. 