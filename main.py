from src.predict import SentimentPredictor

model = SentimentPredictor()

while True:
    text = input("\nEnter review (or 'exit'): ")
    if text.lower() == "exit":
        break

    result = model.predict(text)
    print(result)