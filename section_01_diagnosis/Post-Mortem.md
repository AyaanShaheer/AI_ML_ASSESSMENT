# Post-Mortem Summary (Non-Technical)

We identified three separate issues affecting the chatbot after launch.

**First**, incorrect pricing happened because the chatbot had no live connection to our pricing system. It was generating answers from older training knowledge instead of verified pricing data. We are fixing this by connecting the chatbot to a live pricing source and making it refuse pricing questions when verified data is missing.

**Second**, users writing in Hindi or Arabic were sometimes receiving English replies. This happened because the chatbot’s internal instructions were written in English and were stronger than the user’s language signal. We updated the system prompt so the assistant always responds in the same language as the user.

**Third**, response times increased because the retrieval system slowed down as more documents and users were added. We are improving this by upgrading the search index, reducing unnecessary conversation history, and adding caching.

All **three issues** were caused by system design and infrastructure, not by the AI model itself.