#!/usr/bin/env python3
"""
SMS Feature Extractor

Enhanced SMS feature extraction class with 40 features including:
- Basic text features (length, word count, etc.)
- Advanced pattern detection (URLs, phones, emails)
- Obfuscation detection with multiple patterns
- Risk scoring and weighted features
- Smishing-specific pattern detection

Author: SMS Classification Team
License: MIT
"""

import re
from typing import Dict, List, Any


class SmsFeatureExtractor:
    """
    Enhanced SMS feature extraction class with 40 features including:
    - Basic text features (length, word count, etc.)
    - Advanced pattern detection (URLs, phones, emails)
    - Obfuscation detection with multiple patterns
    - Risk scoring and weighted features
    - Smishing-specific pattern detection
    """
    
    def __init__(self, message: str):
        """Initialize with SMS message."""
        self.message = message
        self.text = message.lower()
        
        # Enhanced keyword lists based on dataset analysis
        self.urgent_keywords = [
            'urgent', 'immediate', 'asap', 'emergency', 'critical', 'important',
            'alert', 'warning', 'notice', 'deadline', 'expire', 'expired',
            'suspended', 'locked', 'blocked', 'compromised', 'breach'
        ]
        
        self.promo_keywords = [
            'free', 'win', 'winner', 'congratulations', 'prize', 'offer',
            'discount', 'sale', 'deal', 'limited', 'exclusive', 'bonus',
            'cash', 'money', 'earn', 'income', 'profit', 'investment'
        ]
        
        self.lottery_keywords = [
            'lottery', 'jackpot', 'million', 'billion', 'winner', 'prize',
            'draw', 'ticket', 'lucky', 'chance', 'opportunity', 'fortune'
        ]
        
        self.action_keywords = [
            'call', 'click', 'reply', 'send', 'text', 'sms', 'visit',
            'download', 'install', 'register', 'sign', 'confirm', 'verify',
            'update', 'unlock', 'restore', 'activate', 'reactivate'
        ]
        
        self.bank_keywords = [
            'bank', 'account', 'card', 'credit', 'debit', 'balance',
            'transaction', 'payment', 'transfer', 'withdraw', 'deposit'
        ]
        
        self.smishing_phrases = [
            'has been suspended', 'to unlock', 'click here', 'security alert',
            'urgent alert', 'account locked', 'verify now', 'confirm immediately'
        ]
        
        # Brand obfuscation patterns
        self.brand_patterns = [
            r'payp[a@]l', r'g[o0][o0]gle', r'@mazon', r'@pple', r'[m]icrosoft',
            r'[f]acebook', r'[t]witter', r'[i]nstagram', r'[w]hatsapp'
        ]
    
    def get_message_length(self) -> int:
        """Get message length in characters."""
        return len(self.message)
    
    def get_word_count(self) -> int:
        """Get word count."""
        return len(self.message.split())
    
    def get_digit_count(self) -> int:
        """Count digits in message."""
        return len(re.findall(r'\d', self.message))
    
    def get_uppercase_ratio(self) -> float:
        """Calculate uppercase character ratio."""
        if not self.message:
            return 0.0
        uppercase_count = sum(1 for c in self.message if c.isupper())
        return uppercase_count / len(self.message)
    
    def get_special_char_count(self) -> int:
        """Count special characters."""
        return len(re.findall(r'[^a-zA-Z0-9\s]', self.message))
    
    def get_url_presence(self) -> int:
        """Check for URL presence."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return 1 if re.search(url_pattern, self.message) else 0
    
    def get_phone_presence(self) -> int:
        """Check for phone number presence."""
        phone_pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        return 1 if re.search(phone_pattern, self.message) else 0
    
    def get_email_presence(self) -> int:
        """Check for email presence."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return 1 if re.search(email_pattern, self.message) else 0
    
    def get_currency_presence(self) -> int:
        """Check for currency symbols."""
        currency_pattern = r'[$£€¥₹]|\b\d+\s*(dollars?|pounds?|euros?|yen|rupees?)\b'
        return 1 if re.search(currency_pattern, self.message, re.IGNORECASE) else 0
    
    def get_obfuscation_presence(self) -> int:
        """Detect obfuscation patterns."""
        obfuscation_patterns = [
            r'[a-z]*[0-9]+[a-z]*',  # Mixed alphanumeric
            r'[a-z]*[!@#$%^&*()]+[a-z]*',  # Special chars in words
            r'\b[a-z]*[A-Z]+[a-z]*\b',  # Mixed case words
            r'[a-z]{1,2}[0-9]{2,}[a-z]{1,2}'  # Number substitution
        ]
        
        for pattern in obfuscation_patterns:
            if re.search(pattern, self.message):
                return 1
        return 0
    
    def get_keyword_count(self, keywords: List[str]) -> int:
        """Count occurrences of keywords."""
        count = 0
        for keyword in keywords:
            count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', self.text))
        return count
    
    def get_urgent_keyword_count(self) -> int:
        """Count urgent keywords."""
        return self.get_keyword_count(self.urgent_keywords)
    
    def get_promo_keyword_count(self) -> int:
        """Count promotional keywords."""
        return self.get_keyword_count(self.promo_keywords)
    
    def get_lottery_keyword_count(self) -> int:
        """Count lottery keywords."""
        return self.get_keyword_count(self.lottery_keywords)
    
    def get_action_keyword_count(self) -> int:
        """Count action keywords."""
        return self.get_keyword_count(self.action_keywords)
    
    def get_bank_keyword_count(self) -> int:
        """Count bank-related keywords."""
        return self.get_keyword_count(self.bank_keywords)
    
    def _detect_obfuscated_keywords(self, keywords: List[str]) -> int:
        """Detect obfuscated keywords using various patterns."""
        count = 0
        for keyword in keywords:
            # Pattern 1: Replace letters with numbers (l33t speak)
            leet_keyword = keyword
            leet_replacements = {
                'a': '[a4@]', 'e': '[e3]', 'i': '[i1!]', 'o': '[o0]',
                's': '[s5$]', 't': '[t7]', 'l': '[l1]', 'g': '[g9]',
                'b': '[b8]', 'z': '[z2]', 'u': '[u]', 'r': '[r]'
            }
            
            for char, replacement in leet_replacements.items():
                leet_keyword = leet_keyword.replace(char, replacement)
            
            if leet_keyword != keyword:
                if re.search(leet_keyword, self.text):
                    count += 1
            
            # Pattern 2: Insert special characters
            special_chars = r'[*\-_\.\s]+'
            obfuscated_pattern = special_chars.join(list(keyword))
            if re.search(obfuscated_pattern, self.text):
                count += 1
        
        return count
    
    def get_obfuscated_urgent_count(self) -> int:
        """Count obfuscated urgent keywords."""
        return self._detect_obfuscated_keywords(self.urgent_keywords)
    
    def get_obfuscated_promo_count(self) -> int:
        """Count obfuscated promotional keywords."""
        return self._detect_obfuscated_keywords(self.promo_keywords)
    
    def get_obfuscated_lottery_count(self) -> int:
        """Count obfuscated lottery keywords."""
        return self._detect_obfuscated_keywords(self.lottery_keywords)
    
    def get_uppercase_word_count(self) -> int:
        """Count words that are entirely uppercase."""
        words = self.message.split()
        return sum(1 for word in words if word.isupper() and len(word) > 1)
    
    def get_all_caps_ratio(self) -> float:
        """Calculate ratio of all-caps words."""
        words = self.message.split()
        if not words:
            return 0.0
        all_caps_count = sum(1 for word in words if word.isupper() and len(word) > 1)
        return all_caps_count / len(words)
    
    def get_suspicious_domain_presence(self) -> int:
        """Check for suspicious domains."""
        suspicious_domains = [
            'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly',
            'short.link', 'is.gd', 'v.gd', 'clck.ru'
        ]
        
        for domain in suspicious_domains:
            if domain in self.text:
                return 1
        return 0
    
    def get_premium_number_presence(self) -> int:
        """Check for premium rate numbers."""
        premium_patterns = [
            r'\b(1900|1800|1300|1200)\d{6}\b',  # Australian premium
            r'\b(900|800|700)\d{7}\b',  # US premium
            r'\b(09\d{2})\d{6}\b'  # UK premium
        ]
        
        for pattern in premium_patterns:
            if re.search(pattern, self.message):
                return 1
        return 0
    
    def get_excessive_punctuation(self) -> int:
        """Detect excessive punctuation."""
        punctuation_count = len(re.findall(r'[!?]{2,}|[.]{3,}', self.message))
        return 1 if punctuation_count > 0 else 0
    
    def get_smishing_phrase_count(self) -> int:
        """Count smishing-specific phrases."""
        count = 0
        for phrase in self.smishing_phrases:
            if phrase in self.text:
                count += 1
        return count
    
    def get_brand_obfuscation_count(self) -> int:
        """Count obfuscated brand names."""
        count = 0
        for pattern in self.brand_patterns:
            if re.search(pattern, self.text):
                count += 1
        return count
    
    def get_composite_features(self) -> Dict[str, int]:
        """Calculate composite features."""
        url_presence = self.get_url_presence()
        phone_presence = self.get_phone_presence()
        urgent_count = self.get_urgent_keyword_count()
        promo_count = self.get_promo_keyword_count()
        bank_count = self.get_bank_keyword_count()
        action_count = self.get_action_keyword_count()
        
        return {
            'url_urgent_combo': url_presence * urgent_count,
            'phone_urgent_combo': phone_presence * urgent_count,
            'url_promo_combo': url_presence * promo_count,
            'bank_action_combo': bank_count * action_count,
            'alert_action_combo': urgent_count * action_count,
            'brand_action_combo': self.get_brand_obfuscation_count() * action_count
        }
    
    def get_all_features(self) -> Dict[str, Any]:
        """Get all 40 features as a dictionary."""
        features = {
            # Basic features
            'message_length': self.get_message_length(),
            'word_count': self.get_word_count(),
            'digit_count': self.get_digit_count(),
            'uppercase_ratio': self.get_uppercase_ratio(),
            'special_char_count': self.get_special_char_count(),
            
            # Pattern detection
            'url_presence': self.get_url_presence(),
            'phone_presence': self.get_phone_presence(),
            'email_presence': self.get_email_presence(),
            'currency_presence': self.get_currency_presence(),
            'obfuscation_presence': self.get_obfuscation_presence(),
            
            # Keyword counts
            'urgent_keyword_count': self.get_urgent_keyword_count(),
            'promo_keyword_count': self.get_promo_keyword_count(),
            'lottery_keyword_count': self.get_lottery_keyword_count(),
            'action_keyword_count': self.get_action_keyword_count(),
            'bank_keyword_count': self.get_bank_keyword_count(),
            
            # Obfuscated keywords
            'obfuscated_urgent_count': self.get_obfuscated_urgent_count(),
            'obfuscated_promo_count': self.get_obfuscated_promo_count(),
            'obfuscated_lottery_count': self.get_obfuscated_lottery_count(),
            
            # Advanced features
            'uppercase_word_count': self.get_uppercase_word_count(),
            'all_caps_ratio': self.get_all_caps_ratio(),
            'suspicious_domain_presence': self.get_suspicious_domain_presence(),
            'premium_number_presence': self.get_premium_number_presence(),
            'excessive_punctuation': self.get_excessive_punctuation(),
            
            # Smishing-specific
            'smishing_phrase_count': self.get_smishing_phrase_count(),
            'brand_obfuscation_count': self.get_brand_obfuscation_count(),
        }
        
        # Add composite features
        features.update(self.get_composite_features())
        
        # Calculate risk score
        risk_factors = [
            features['url_presence'] * 3,
            features['phone_presence'] * 2,
            features['urgent_keyword_count'] * 2,
            features['obfuscation_presence'] * 2,
            features['suspicious_domain_presence'] * 4,
            features['smishing_phrase_count'] * 3,
            features['brand_obfuscation_count'] * 2
        ]
        features['risk_score'] = sum(risk_factors)
        
        return features
    
    def get_feature_vector(self) -> List[float]:
        """Get feature vector for ML models."""
        features = self.get_all_features()
        # Return only the numerical features (exclude risk_score for now)
        feature_names = [
            'message_length', 'word_count', 'digit_count', 'uppercase_ratio',
            'special_char_count', 'url_presence', 'phone_presence', 'email_presence',
            'currency_presence', 'obfuscation_presence', 'urgent_keyword_count',
            'promo_keyword_count', 'lottery_keyword_count', 'action_keyword_count',
            'bank_keyword_count', 'obfuscated_urgent_count', 'obfuscated_promo_count',
            'obfuscated_lottery_count', 'uppercase_word_count', 'all_caps_ratio',
            'suspicious_domain_presence', 'premium_number_presence', 'excessive_punctuation',
            'url_urgent_combo', 'phone_urgent_combo', 'url_promo_combo',
            'smishing_phrase_count', 'brand_obfuscation_count', 'bank_action_combo',
            'alert_action_combo', 'brand_action_combo'
        ]
        return [features[name] for name in feature_names]
