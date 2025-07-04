#!/usr/bin/env python
# coding: utf-8

# ##Vishnu Koushik Tekuru, koushiktekuru@gmail.com. HMLR 2025 Data Scientist Challenge

# In[3]:
#pip install --upgrade pip
#pip install nltk

import os
import re
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import pickle
from datetime import datetime

# NLP and ML libraries 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans


# In[7]:


import nltk

def download_nltk_data():
    """Download all NLTK data"""
    try:
        print("Downloading all NLTK data (this may take a while)...")
        nltk.download('all')
        print("All NLTK data downloaded successfully")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

# Download data at import time
download_nltk_data()


# In[5]:


class BBCAnalyzer:
    """
    BBC dataset analyzer
    Features: Better NLP, multiple classifiers, clustering, proper evaluation
    """
    
    def __init__(self, dataset_path="bbc"):
        self.dataset_path = dataset_path
        self.documents = []
        self.labels = []
        self.processed_documents = []
        
        # Initialize NLP tools with error handling
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            print("✓ NLP tools initialized successfully")
        except Exception as e:
            print(f"✗ Error initializing NLP tools: {e}")
            # Fallback to basic stop words
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            self.lemmatizer = None
        
        # Enhanced keyword dictionaries with more sophisticated patterns
        self.subcategory_keywords = {
            'business': {
                'stock_market': {
                    'primary': ['stock', 'share', 'market', 'trading', 'exchange'],
                    'secondary': ['nasdaq', 'dow', 'ftse', 'index', 'equity', 'portfolio']
                },
                'company_news': {
                    'primary': ['company', 'firm', 'corporation', 'business'],
                    'secondary': ['ceo', 'executive', 'board', 'corporate', 'management']
                },
                'mergers_acquisitions': {
                    'primary': ['merger', 'acquisition', 'takeover', 'buyout'],
                    'secondary': ['deal', 'acquire', 'merge', 'purchase', 'consolidation']
                },
                'financial_results': {
                    'primary': ['profit', 'revenue', 'earnings', 'financial'],
                    'secondary': ['quarterly', 'annual', 'results', 'income', 'turnover']
                },
                'economic_policy': {
                    'primary': ['economy', 'economic', 'policy', 'government'],
                    'secondary': ['inflation', 'interest', 'budget', 'fiscal', 'monetary']
                }
            },
            'entertainment': {
                'cinema': {
                    'primary': ['film', 'movie', 'cinema', 'hollywood'],
                    'secondary': ['actor', 'actress', 'director', 'oscar', 'premiere', 'box office']
                },
                'television': {
                    'primary': ['television', 'tv', 'show', 'series'],
                    'secondary': ['channel', 'broadcast', 'programme', 'episode', 'season']
                },
                'music': {
                    'primary': ['music', 'song', 'album', 'singer'],
                    'secondary': ['band', 'concert', 'tour', 'chart', 'record', 'artist']
                },
                'theatre': {
                    'primary': ['theatre', 'theater', 'play', 'stage'],
                    'secondary': ['west end', 'broadway', 'performance', 'drama', 'musical']
                },
                'celebrity_news': {
                    'primary': ['celebrity', 'star', 'famous', 'personality'],
                    'secondary': ['gossip', 'scandal', 'relationship', 'marriage', 'divorce']
                }
            },
            'sport': {
                'football': {
                    'primary': ['football', 'soccer', 'premier league', 'fifa'],
                    'secondary': ['goal', 'match', 'team', 'player', 'manager', 'transfer']
                },
                'cricket': {
                    'primary': ['cricket', 'test', 'wicket', 'bat'],
                    'secondary': ['bowl', 'ashes', 'county', 'icc', 'innings', 'runs']
                },
                'tennis': {
                    'primary': ['tennis', 'wimbledon', 'court', 'serve'],
                    'secondary': ['match', 'set', 'game', 'tournament', 'ranking']
                },
                'olympics': {
                    'primary': ['olympic', 'olympics', 'medal', 'gold'],
                    'secondary': ['silver', 'bronze', 'athlete', 'games', 'torch']
                },
                'rugby': {
                    'primary': ['rugby', 'scrum', 'try', 'union'],
                    'secondary': ['league', 'world cup', 'six nations', 'tackle']
                }
            }
        }
        
        # Media personality identification patterns
        self.personality_indicators = {
            'politician': {
                'titles': ['minister', 'mp', 'president', 'senator', 'mayor', 'councillor'],
                'contexts': ['government', 'parliament', 'election', 'policy', 'cabinet']
            },
            'actor': {
                'titles': ['actor', 'actress', 'star', 'performer'],
                'contexts': ['film', 'movie', 'cinema', 'role', 'character', 'performance']
            },
            'musician': {
                'titles': ['singer', 'musician', 'artist', 'performer'],
                'contexts': ['music', 'song', 'album', 'concert', 'tour', 'band']
            },
            'athlete': {
                'titles': ['player', 'athlete', 'captain', 'striker', 'goalkeeper'],
                'contexts': ['sport', 'team', 'match', 'game', 'tournament', 'championship']
            },
            'tv_personality': {
                'titles': ['presenter', 'host', 'anchor', 'reporter'],
                'contexts': ['television', 'show', 'programme', 'broadcast', 'channel']
            }
        }
    
    def load_dataset(self):
        """Load BBC dataset with error handling and statistics"""
        print("Loading BBC dataset...")
        
        categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
        load_stats = {}
        
        for category in categories:
            category_path = os.path.join(self.dataset_path, category)
            category_count = 0
            
            if os.path.exists(category_path):
                for filename in os.listdir(category_path):
                    if filename.endswith('.txt'):
                        file_path = os.path.join(category_path, filename)
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                                content = file.read().strip()
                                if len(content) > 50:  # Only keep substantial articles
                                    self.documents.append(content)
                                    self.labels.append(category)
                                    category_count += 1
                        except Exception as e:
                            print(f"Warning: Could not read {filename}: {e}")
                
                load_stats[category] = category_count
            else:
                print(f"Warning: Category folder '{category}' not found")
                load_stats[category] = 0
        
        print(f"\nDataset loaded successfully!")
        print(f"Total articles: {len(self.documents)}")
        print("\nArticles per category:")
        for category, count in load_stats.items():
            print(f"  {category}: {count} articles")
        
        return len(self.documents) > 0
    
    def preprocess_text(self, text, remove_stopwords=True, lemmatize=True):
        """
        Enhanced text preprocessing with error handling
        """
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters but keep sentence structure
            text = re.sub(r'[^a-zA-Z\s\.]', ' ', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords if requested
            if remove_stopwords:
                tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
            
            # Lemmatize if requested and available
            if lemmatize and self.lemmatizer:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            return ' '.join(tokens)
        except Exception as e:
            print(f"Error preprocessing text: {e}")
            # Return basic cleaned text
            return re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    
    def subcategorization(self):
        """
        Enhanced sub-categorization using weighted keyword matching
        """
        print("\n=== SUB-CATEGORIZATION ===")
        
        results = {}
        
        for main_category in ['business', 'entertainment', 'sport']:
            if main_category not in self.subcategory_keywords:
                continue
                
            # Get articles for this category
            category_articles = [(i, doc) for i, doc in enumerate(self.documents) 
                               if self.labels[i] == main_category]
            
            if not category_articles:
                continue
            
            print(f"\nAnalyzing {main_category} articles...")
            subcategory_scores = defaultdict(list)
            
            for article_idx, article in category_articles:
                article_lower = article.lower()
                article_scores = {}
                
                # Calculate weighted scores for each subcategory
                for subcategory, keywords in self.subcategory_keywords[main_category].items():
                    primary_score = sum(2 for keyword in keywords['primary'] 
                                      if keyword in article_lower)
                    secondary_score = sum(1 for keyword in keywords['secondary'] 
                                        if keyword in article_lower)
                    
                    total_score = primary_score + secondary_score
                    article_scores[subcategory] = total_score
                
                # Assign to best matching subcategory
                if article_scores:
                    best_subcategory = max(article_scores, key=article_scores.get)
                    if article_scores[best_subcategory] > 0:
                        subcategory_scores[best_subcategory].append({
                            'article_idx': article_idx,
                            'score': article_scores[best_subcategory],
                            'preview': article[:200] + "..."
                        })
            
            results[main_category] = dict(subcategory_scores)
            
            # Display results
            print(f"\n{main_category.upper()} Sub-categories:")
            for subcategory, articles in subcategory_scores.items():
                print(f"  {subcategory}: {len(articles)} articles")
                if articles:
                    best_article = max(articles, key=lambda x: x['score'])
                    print(f"    Best match (score: {best_article['score']}): {best_article['preview']}")
        
        return results
    
    def extract_named_entities(self):
        """
        Extract named entities using NLTK with improved error handling
        """
        print("\n=== NAMED ENTITY EXTRACTION ===")
        
        all_entities = {'persons': [], 'organizations': [], 'locations': []}
        media_personalities = defaultdict(list)
        
        # Process a sample of documents to avoid overwhelming output
        sample_size = min(100, len(self.documents))
        sample_indices = np.random.choice(len(self.documents), sample_size, replace=False)
        
        print(f"Processing {sample_size} articles for named entities...")
        
        for idx in sample_indices:
            document = self.documents[idx]
            category = self.labels[idx]
            
            try:
                # Process first few sentences for efficiency
                sentences = sent_tokenize(document)[:3]
                
                for sentence in sentences:
                    try:
                        # Get named entities
                        tokens = word_tokenize(sentence)
                        pos_tags = pos_tag(tokens)
                        chunks = ne_chunk(pos_tags)
                        
                        for chunk in chunks:
                            if hasattr(chunk, 'label'):
                                entity_name = ' '.join([token for token, pos in chunk.leaves()])
                                
                                if chunk.label() == 'PERSON':
                                    all_entities['persons'].append(entity_name)
                                    
                                    # Try to classify this person
                                    personality_type = self._classify_personality(sentence, entity_name)
                                    if personality_type:
                                        media_personalities[personality_type].append({
                                            'name': entity_name,
                                            'category': category,
                                            'context': sentence,
                                            'confidence': self._calculate_confidence(sentence, personality_type)
                                        })
                                
                                elif chunk.label() == 'ORGANIZATION':
                                    all_entities['organizations'].append(entity_name)
                                
                                elif chunk.label() in ['GPE', 'GSP']:  # Geographic entities
                                    all_entities['locations'].append(entity_name)
                    
                    except Exception as e:
                        print(f"Error processing sentence for NER: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error processing document {idx} for NER: {e}")
                continue
        
        # Clean up and count entities
        for entity_type in all_entities:
            all_entities[entity_type] = list(set(all_entities[entity_type]))
        
        # Display results
        print(f"\nEntities found:")
        for entity_type, entities in all_entities.items():
            print(f"  {entity_type}: {len(entities)} unique entities")
            if entities:
                print(f"    Examples: {', '.join(entities[:5])}")
        
        print(f"\nMedia personalities by type:")
        for personality_type, people in media_personalities.items():
            unique_people = {}
            for person in people:
                if person['name'] not in unique_people or person['confidence'] > unique_people[person['name']]['confidence']:
                    unique_people[person['name']] = person
            
            print(f"  {personality_type}: {len(unique_people)} people")
            for name, info in list(unique_people.items())[:5]:
                print(f"    - {name} (confidence: {info['confidence']:.2f})")
        
        return all_entities, dict(media_personalities)
    
    def _classify_personality(self, sentence, person_name):
        """Classify a person's role based on context"""
        sentence_lower = sentence.lower()
        
        best_match = None
        best_score = 0
        
        for personality_type, indicators in self.personality_indicators.items():
            score = 0
            
            # Check for titles
            for title in indicators['titles']:
                if title in sentence_lower:
                    score += 3
            
            # Check for context words
            for context in indicators['contexts']:
                if context in sentence_lower:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = personality_type
        
        return best_match if best_score > 0 else None
    
    def _calculate_confidence(self, sentence, personality_type):
        """Calculate confidence score for personality classification"""
        if personality_type not in self.personality_indicators:
            return 0.0
        
        sentence_lower = sentence.lower()
        indicators = self.personality_indicators[personality_type]
        
        title_matches = sum(1 for title in indicators['titles'] if title in sentence_lower)
        context_matches = sum(1 for context in indicators['contexts'] if context in sentence_lower)
        
        # Normalize confidence score
        max_possible = len(indicators['titles']) + len(indicators['contexts'])
        actual_score = (title_matches * 3) + context_matches
        
        return min(actual_score / max_possible, 1.0)
    
    def extract_april_events(self):
        """
        Extract April events with improved pattern matching and context
        """
        print("\n=== APRIL EVENTS EXTRACTION ===")
        
        april_events = []
        
        # More sophisticated April patterns
        april_patterns = [
            r'april\s+\d{1,2}(?:st|nd|rd|th)?',
            r'\d{1,2}(?:st|nd|rd|th)?\s+april',
            r'in\s+april',
            r'during\s+april',
            r'throughout\s+april',
            r'april\s+\d{4}',
            r'early\s+april',
            r'late\s+april',
            r'mid-april'
        ]
        
        for i, document in enumerate(self.documents):
            try:
                sentences = sent_tokenize(document)
                
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    
                    # Check if sentence contains April reference
                    april_match = None
                    for pattern in april_patterns:
                        match = re.search(pattern, sentence_lower)
                        if match:
                            april_match = match.group()
                            break
                    
                    if april_match:
                        # Get surrounding context
                        sentence_idx = sentences.index(sentence)
                        context_start = max(0, sentence_idx - 1)
                        context_end = min(len(sentences), sentence_idx + 2)
                        context = ' '.join(sentences[context_start:context_end])
                        
                        # Try to identify event type
                        event_type = self._identify_event_type(sentence_lower)
                        
                        april_events.append({
                            'article_index': i,
                            'category': self.labels[i],
                            'april_reference': april_match,
                            'event_type': event_type,
                            'sentence': sentence,
                            'context': context,
                            'summary': self._generate_event_summary(sentence)
                        })
            except Exception as e:
                print(f"Error processing document {i} for April events: {e}")
                continue
        
        # Group by category and event type
        events_by_category = defaultdict(list)
        events_by_type = defaultdict(list)
        
        for event in april_events:
            events_by_category[event['category']].append(event)
            if event['event_type']:
                events_by_type[event['event_type']].append(event)
        
        print(f"Found {len(april_events)} April event references")
        
        print(f"\nBy category:")
        for category, events in events_by_category.items():
            print(f"  {category}: {len(events)} events")
        
        print(f"\nBy event type:")
        for event_type, events in events_by_type.items():
            print(f"  {event_type}: {len(events)} events")
            if events:
                print(f"    Example: {events[0]['summary']}")
        
        return april_events
    
    def _identify_event_type(self, sentence):
        """Identify the type of event mentioned"""
        event_keywords = {
            'meeting': ['meeting', 'conference', 'summit', 'gathering'],
            'launch': ['launch', 'release', 'unveil', 'debut', 'premiere'],
            'competition': ['tournament', 'championship', 'competition', 'match', 'game'],
            'announcement': ['announce', 'reveal', 'declare', 'statement'],
            'performance': ['concert', 'show', 'performance', 'tour', 'festival']
        }
        
        for event_type, keywords in event_keywords.items():
            if any(keyword in sentence for keyword in keywords):
                return event_type
        
        return 'other'
    
    def _generate_event_summary(self, sentence):
        """Generate a concise summary of the event"""
        # Simple extractive summary - take first 100 characters
        summary = sentence.strip()
        if len(summary) > 100:
            summary = summary[:100] + "..."
        return summary
    
    def train_multiple_classifiers(self):
        """
        Train and compare multiple classifiers with error handling
        """
        print("\n=== TRAINING MULTIPLE CLASSIFIERS ===")
        
        # Preprocess all documents
        print("Preprocessing documents...")
        self.processed_documents = [self.preprocess_text(doc) for doc in self.documents]
        
        # Create feature vectors using TF-IDF
        print("Creating feature vectors...")
        vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X = vectorizer.fit_transform(self.processed_documents)
        y = self.labels
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define classifiers to try
        classifiers = {
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(kernel='linear', random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        best_classifier = None
        best_score = 0
        
        print("\nTraining and evaluating classifiers...")
        
        for name, clf in classifiers.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train classifier
                clf.fit(X_train, y_train)
                
                # Cross-validation score
                cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
                
                # Test score
                test_score = clf.score(X_test, y_test)
                
                # Predictions for detailed analysis
                y_pred = clf.predict(X_test)
                
                results[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_score': test_score,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                print(f"  Test Score: {test_score:.3f}")
                
                if test_score > best_score:
                    best_score = test_score
                    best_classifier = clf
                    
            except Exception as e:
                print(f"  Error training {name}: {e}")
                continue
        
        # Save best classifier and vectorizer
        self.best_classifier = best_classifier
        self.vectorizer = vectorizer
        
        if best_classifier:
            print(f"\nBest classifier: {type(best_classifier).__name__} with score: {best_score:.3f}")
        else:
            print("\nNo classifier trained successfully!")
        
        return results, best_classifier, vectorizer
    
    def cluster_documents(self, n_clusters=10):
        """
        Perform document clustering to find hidden patterns
        """
        print(f"\n=== DOCUMENT CLUSTERING (k={n_clusters}) ===")
        
        if not hasattr(self, 'vectorizer'):
            print("Training vectorizer first...")
            self.train_multiple_classifiers()
        
        # Use existing vectorizer
        X = self.vectorizer.transform(self.processed_documents)
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Analyze clusters
        cluster_analysis = defaultdict(list)
        
        for i, cluster_id in enumerate(clusters):
            cluster_analysis[cluster_id].append({
                'doc_index': i,
                'category': self.labels[i],
                'preview': self.documents[i][:150] + "..."
            })
        
        print(f"Documents clustered into {n_clusters} groups:")
        
        for cluster_id, docs in cluster_analysis.items():
            categories = [doc['category'] for doc in docs]
            category_dist = Counter(categories)
            
            print(f"\nCluster {cluster_id}: {len(docs)} documents")
            print(f"  Category distribution: {dict(category_dist)}")
            
            if docs:
                print(f"  Example: {docs[0]['preview']}")
        
        return cluster_analysis
    
    def run_complete_analysis(self):
        """
        Run the complete analysis with error handling
        """
        print("=" * 60)
        print("BBC DATASET ANALYSIS")
        print("=" * 60)
        
        try:
            # Step 1: Load dataset
            if not self.load_dataset():
                print("Failed to load dataset. Please check the path and structure.")
                return None
            
            # Step 2: Sub-categorization
            subcategories = self.subcategorization()
            
            # Step 3: Named entity extraction
            entities, personalities = self.extract_named_entities()
            
            # Step 4: April events extraction
            april_events = self.extract_april_events()
            
            # Step 5: Train multiple classifiers
            classifier_results, best_classifier, vectorizer = self.train_multiple_classifiers()
            
            # Step 6: Document clustering
            clusters = self.cluster_documents()
            
            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETE!")
            print("=" * 60)
            
            # Compile results
            results = {
                'dataset_stats': dict(Counter(self.labels)),
                'subcategories': subcategories,
                'entities': entities,
                'media_personalities': personalities,
                'april_events': april_events,
                'classifier_results': classifier_results,
                'best_classifier': best_classifier,
                'vectorizer': vectorizer,
                'document_clusters': clusters
            }
            
            return results
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_results(self, results, filename='bbc_analysis_results.pkl'):
        """Save results to file"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def classify_new_text(self, text):
        """Classify new text using the trained model"""
        if not hasattr(self, 'best_classifier') or not hasattr(self, 'vectorizer'):
            print("Please train the classifier first!")
            return None
        
        try:
            processed_text = self.preprocess_text(text)
            text_vector = self.vectorizer.transform([processed_text])
            
            prediction = self.best_classifier.predict(text_vector)[0]
            probabilities = self.best_classifier.predict_proba(text_vector)[0]
            
            result = {
                'prediction': prediction,
                'probabilities': dict(zip(self.best_classifier.classes_, probabilities))
            }
            
            return result
        except Exception as e:
            print(f"Error classifying text: {e}")
            return None


# In[8]:


def main():
    """Main execution function"""
    
    try:
        # Initialize analyzer
        analyzer = BBCAnalyzer(r"C:\Users\koush\Desktop\BBC\bbc") # change the path accordingly
        
        # Run complete analysis
        results = analyzer.run_complete_analysis()
        
        if results:
            # Save results
            analyzer.save_results(results)
            
            # Demonstrate classifier
            print("\n" + "=" * 40)
            print("CLASSIFIER DEMONSTRATION")
            print("=" * 40)
            
            test_texts = [
                "The company reported strong quarterly earnings with revenue up 15% year-over-year.",
                "The new Marvel movie broke box office records in its opening weekend.",
                "Manchester United signed a new striker for £50 million in the summer transfer window.",
                "The Prime Minister announced new environmental policies in Parliament today.",
                "Apple unveiled its latest iPhone with improved camera technology and longer battery life."
            ]
            
            for text in test_texts:
                result = analyzer.classify_new_text(text)
                if result:
                    print(f"\nText: {text}")
                    print(f"Predicted Category: {result['prediction']}")
                    print("Probabilities:")
                    for category, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                        print(f"  {category}: {prob:.3f}")
            
            return results
        else:
            print("Analysis failed. Please check your dataset.")
            return None
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        return None

if __name__ == "__main__":
    results = main()


# In[ ]:




