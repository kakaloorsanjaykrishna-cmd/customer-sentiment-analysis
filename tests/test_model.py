from src.predict import SentimentPredictor

def test_prediction():
    model = SentimentPredictor()
    result = model.predict("This product is amazing")
    
    assert "label" in result
    assert "confidence" in result