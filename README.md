# 🚀 Hybrid SMS Classification System

A production-ready, high-accuracy SMS classification system that combines feature-based machine learning with n-gram text analysis to classify SMS messages as **ham**, **smishing**, or **spam**.

## ✨ Features

- **🎯 High Accuracy**: 93.9% accuracy on test data (88.1% Stage 1 with weighted features)
- **🔀 Two-Stage Classification**: Intelligent uncertainty detection
- **📊 38 Enhanced Features**: Advanced pattern detection with strategic weighting
- **⚡ Real-time Performance**: ~0.7ms average processing time
- **🛡️ Security Focus**: Excellent smishing detection (85.0% F1-score)
- **🔧 Production Ready**: Clean API, comprehensive testing, detailed documentation

## 🏗️ Architecture

### Modular Design
The system is split into two main modules:
- **`sms_feature_extractor.py`**: Standalone feature extraction with 38 enhanced features
- **`sms_classifier_hybrid.py`**: Complete hybrid classification system

### Stage 1: Feature-Based Classification
- **38 Enhanced Features**: 31 base features + 7 weighted features for optimal performance
- **Weighted Features**: Strategic weighting of high-impact features (URL 3x, phone 2x, etc.)
- **Risk Scoring**: Combined weighted features for threat assessment
- **Smishing Detection**: Bank keywords, action words, alert phrases
- **Obfuscation Detection**: L33t speak, special character insertion, brand obfuscation

### Stage 2: N-gram + ML Algorithms
- **TfidfVectorizer**: 1-2 gram text vectorization with 5000 features
- **Ensemble Methods**: Naive Bayes + Logistic Regression + Decision Tree
- **Voting System**: Majority vote for final prediction

### Hybrid Combination
- **Intelligent Stage Selection**: Uses Stage 2 for uncertain cases (confidence < 0.90)
- **Optimal Performance**: Balanced accuracy with efficient Stage 2 usage
- **Dynamic Processing**: Fast Stage 1 for clear cases, detailed Stage 2 for uncertain ones

## 📋 Dataset Format

Your CSV file should have the following columns:
- `TEXT`: SMS message content
- `LABEL`: Target class (`ham`, `smishing`, or `spam`)

Example:
```csv
TEXT,LABEL
"Hey! Want to meet for coffee?",ham
"URGENT: Your account has been suspended. Click here to unlock: http://fake-bank.com",smishing
"Congratulations! You've won $1000! Click here to claim your prize!",spam
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/hybrid-sms-classifier.git
cd hybrid-sms-classifier
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Train the model**:
```python
from sms_classifier_hybrid import HybridSmsClassifier

# Initialize and train
classifier = HybridSmsClassifier()
classifier.train("dataset.csv")

# Classify a message
result = classifier.classify("URGENT: Your account has been suspended!")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Alternative: Use Feature Extractor Separately
```python
from sms_feature_extractor import SmsFeatureExtractor

# Extract features from a message
sms = SmsFeatureExtractor("URGENT: Your account has been suspended!")
features = sms.get_all_features()
print(f"Risk score: {features['risk_score']}")
print(f"URL presence: {features['url_presence']}")
```

### Run Complete Pipeline

```bash
python3 sms_classifier_hybrid.py
```

This will:
1. Train both stages on 80% of your data
2. Evaluate on the remaining 20%
3. Show detailed performance metrics
4. Display confusion matrix and classification report

## 🔧 API Reference

### HybridSmsClassifier

#### `train(csv_path: str, test_size: float = 0.2)`
Train the complete hybrid system on your dataset with proper train/test split.
- **80%** of data used for training
- **20%** of data held out for testing
- Both stages trained only on training data

#### `classify(message: str) -> Dict`
Classify a single SMS message.

**Returns:**
```python
{
    'prediction': 'ham|smishing|spam',  # Final prediction
    'confidence': 0.95,  # Confidence score
    'probabilities': {'ham': 0.95, 'smishing': 0.03, 'spam': 0.02},  # Class probabilities
    'stage_used': 'Stage 1|Stage 2',  # Which stage made the final decision
    'stage1_prediction': 'ham',  # Stage 1 prediction
    'stage1_confidence': 0.85,  # Stage 1 confidence
    'stage1_probabilities': {'ham': 0.85, 'smishing': 0.10, 'spam': 0.05},
    'stage2_prediction': 'ham',  # Stage 2 prediction (if used)
    'stage2_confidence': 0.95,  # Stage 2 confidence (if used)
    'stage2_probabilities': {'ham': 0.95, 'smishing': 0.03, 'spam': 0.02},  # (if used)
    'processing_time_ms': 0.7,  # Processing time in milliseconds
    'features': {...}  # All 40 extracted features
}
```

#### `evaluate() -> Dict`
Evaluate the system on held-out test data (no parameters needed).

**Returns:**
```python
{
    'combined_accuracy': 0.945,   # Combined system accuracy on test set
    'stage1_accuracy': 0.870,     # Stage 1 individual accuracy on test set
    'stage2_accuracy': 0.939,     # Stage 2 individual accuracy on test set
    'stage1_correct': 1061,       # Stage 1 correct predictions on test set
    'stage2_correct': 978,        # Stage 2 correct predictions on test set
    'combined_correct': 1153,     # Combined correct predictions on test set
    'total_messages': 1220,       # Size of test set (20% of total)
    'stage1_only_count': 179,     # Messages handled by Stage 1 only
    'stage1_only_percentage': 14.7, # Percentage handled by Stage 1 only
    'stage2_usage_count': 1041,   # Messages using Stage 2
    'stage2_usage_percentage': 85.3, # Percentage using Stage 2
    'avg_processing_time_ms': 0.7,
    'results': [...]  # Detailed results for each test message
}
```

### SmsFeatureExtractor

The `SmsFeatureExtractor` class is available in a separate module for standalone use.

#### `get_all_features() -> Dict`
Extract all 40 features from an SMS message.

#### `get_feature_vector() -> List[float]`
Get feature vector for ML models.

#### `get_message_length() -> int`
Get message length in characters.

#### `get_risk_score() -> int`
Calculate risk score based on multiple factors.

#### `get_obfuscation_presence() -> int`
Detect obfuscation patterns in the message.

## 📈 Performance Metrics

### Overall Performance
- **Combined System Accuracy**: 93.9%
- **Stage 1 Individual Accuracy**: 88.1% (with weighted features)
- **Stage 2 Individual Accuracy**: 93.3%
- **Improvement over Stage 1**: +5.7%
- **Average Processing Time**: 0.7ms

### Class-Specific Performance
- **Ham**: 96.0% precision, 100.0% recall, 98.0% F1
- **Smishing**: 90.0% precision, 81.0% recall, 85.0% F1
- **Spam**: 73.0% precision, 56.0% recall, 64.0% F1

### Stage Usage Statistics
- **Stage 1 handles**: 14.7% of messages (high confidence cases)
- **Stage 2 used for**: 85.3% of messages (uncertain cases)

## 🎯 Stage Selection Logic

The system uses intelligent stage selection based on confidence thresholds:

```python
def should_use_stage2(self, prediction: str, confidence: float, probabilities: np.ndarray) -> bool:
    """Use Stage 2 for uncertain cases"""
    return confidence < 0.90
```

**Logic:**
- **Stage Selection**: Use Stage 2 if confidence < 0.90
- **High confidence cases**: Use Stage 1 only (faster processing)
- **Uncertain cases**: Use Stage 2 for more detailed analysis

## 📊 Feature Categories

### Basic Features (5)
- Message length, word count, digit count, uppercase ratio, special characters

### Pattern Detection (5)
- URL presence, phone number presence, email presence, currency presence, obfuscation presence

### Keyword Analysis (10)
- Urgent keywords, promo keywords, lottery keywords, action keywords, bank keywords
- Obfuscated versions of urgent, promo, and lottery keywords

### Advanced Features (10)
- Uppercase word count, all-caps ratio, suspicious domain presence, premium number presence
- Excessive punctuation, smishing phrases, brand obfuscation detection

### Composite Features (10)
- URL + urgent combo, phone + urgent combo, URL + promo combo
- Bank + action combo, alert + action combo, brand + action combo

## 🧪 Testing

Run the test suite to verify everything works correctly:

```bash
python3 test_classifier.py
```

Expected output:
```
Running Hybrid SMS Classification System Tests
============================================================
Testing SMS Feature Extraction...
✅ Feature extraction test passed!

Testing Stage 2 Classifier...
✅ Stage 2 classifier test passed!

Testing Classifier Initialization...
✅ Classifier initialization test passed!

Testing Sample Classification...
✅ Sample classification test passed!

Test Results: 4/4 tests passed
🎉 All tests passed! The system is working correctly.
```

## 📝 Example Usage

### Basic Classification
```python
from sms_classifier_hybrid import HybridSmsClassifier

# Initialize and train
classifier = HybridSmsClassifier()
classifier.train("dataset.csv")

# Classify messages
messages = [
    "Hey! Want to meet for coffee?",  # Ham
    "URGENT: Account suspended. Click here!",  # Smishing
    "Win $1000! Click now!",  # Spam
]

for message in messages:
    result = classifier.classify(message)
    print(f"Message: {message}")
    print(f"Combined Prediction: {result['prediction']}")
    print(f"Combined Confidence: {result['confidence']:.3f}")
    print(f"Stage 1: {result['stage1_prediction']} (conf: {result['stage1_confidence']:.3f})")
    print(f"Stage 2: {result['stage2_prediction']} (conf: {result['stage2_confidence']:.3f})")
    print(f"Weights: Stage1={result['stage1_weight']:.3f}, Stage2={result['stage2_weight']:.3f}")
    print(f"Risk Score: {result['features']['risk_score']}")
    print("-" * 50)
```

### Evaluation and Analysis
```python
# Evaluate on test data
evaluation = classifier.evaluate()

print(f"Overall Accuracy: {evaluation['combined_accuracy']:.1%}")
print(f"Stage 1 Accuracy: {evaluation['stage1_accuracy']:.1%}")
print(f"Stage 2 Accuracy: {evaluation['stage2_accuracy']:.1%}")
print(f"Stage 2 Usage: {evaluation['stage2_usage_percentage']:.1f}%")
```

## 🔬 Research and Development

This system was developed through extensive research and testing:

- **Feature Engineering**: 40 carefully crafted features based on SMS analysis
- **Threshold Optimization**: Tested confidence thresholds from 0.5 to 0.9
- **Stage Selection**: Analyzed different conditions for Stage 2 usage
- **Performance Tuning**: Optimized for both accuracy and efficiency

### Key Findings
- **Best Configuration**: `all_predictions` with confidence < 0.9
- **Optimal Stage 2 Usage**: 85.3% provides best accuracy/efficiency balance
- **Feature Importance**: URL presence, urgent keywords, and obfuscation detection are most effective
- **Class Imbalance**: SMOTE balancing significantly improves minority class performance

## 📄 License

MIT License - feel free to use this project for research, commercial, or educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📞 Support

If you encounter any issues or have questions:
1. Check the test suite: `python3 test_classifier.py`
2. Review the example usage in `example.py`
3. Open an issue on GitHub

## 🎉 Acknowledgments

- Built with scikit-learn, pandas, and numpy
- Uses imbalanced-learn for SMOTE oversampling
- Inspired by research in SMS security and text classification

---

**Ready to classify SMS messages with 94.5% accuracy? Get started now!** 🚀