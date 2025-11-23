import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class CircumplexMapping:
    """
    This class maps Empath feature vectors into a Valence-Arousal (Circumplex) space
    using NRC-VAD or VADER as a fallback.
    """

    def __init__(self, vad_path=None):
        self.vader = SentimentIntensityAnalyzer()
        self.vad_lexicon = self._load_nrc_vad(vad_path)
        self.empath_circumplex_map = self._create_empath_circumplex_mapping()
        self.cache = {}

    def _load_nrc_vad(self, file_path):
        """Load NRC-VAD lexicon and normalize valence/arousal to [-1, 1]."""
        if not file_path or not os.path.exists(file_path):
            print("NRC-VAD not loaded (missing file path).")
            return {}

        df = pd.read_csv(file_path, sep='\t')
        lexicon = {}
        for _, row in df.iterrows():
            if pd.isna(row['term']):
                continue
            
            word = str(row['term']).strip().lower()
            if word: 
                valence = (row['valence'] - 0.5) * 2
                arousal = (row['arousal'] - 0.5) * 2
                lexicon[word] = {'valence': valence, 'arousal': arousal}
        
        print(f"Loaded NRC-VAD: {len(lexicon)} terms")
        return lexicon

    def _create_empath_circumplex_mapping(self):
        """
        Create mapping from Empath categories to circumplex coordinates
        """
        mapping = {}
        
        if hasattr(self, 'nrc_vad') and self.vad_lexicon:
            emotion_words = [
            'joy', 'excitement', 'anticipation', 'optimism', 'achievement', 'celebration',
            'enthusiasm', 'confidence', 'pride', 'contentment', 'pleasure', 'love',
            'trust', 'warmth', 'calm', 'peaceful', 'relaxed', 'serene', 'acceptance',
            'hope', 'comfort', 'gentle', 'quiet', 'sadness', 'depression', 'melancholy',
            'disappointment', 'loneliness', 'emptiness', 'hopelessness', 'despair',
            'grief', 'sorrow', 'gloom', 'anger', 'fear', 'anxiety', 'rage', 'panic',
            'frustration', 'irritability', 'nervousness', 'terror', 'horror',
            'agitation', 'stress', 'tension', 'pain', 'suffering', 'exhaustion',
            'fatigue', 'confusion', 'clarity', 'focus', 'memory', 'concentration',
            'aggression', 'violence', 'control', 'discipline', 'family', 'friends',
            'relationship', 'communication', 'connection', 'isolation', 'rejection',
            'abandonment', 'betrayal', 'health', 'medical', 'therapy', 'healing',
            'recovery', 'support', 'help', 'medication', 'treatment'
        ]
            for word in emotion_words:
                if word in self.vad_lexicon:
                    mapping[word] = (self.vad_lexicon[word]['valence'], self.vad_lexicon[word]['arousal'])

        empath_fallbacks = {
        'positive emotion': (0.6, 0.1),  
        'optimism': (0.5, 0.2),
        'love': (0.7, 0.2),            
        'cheerfulness': (0.6, 0.3),
        'joy': (0.7, 0.4),
        'pride': (0.5, 0.2),
        'contentment': (0.4, -0.3),          
        'happiness': (0.6, 0.3),           
        'excitement': (0.5, 0.6),          
        
        # Negative emotions
        'negative emotion': (-0.6, 0.0),    
        'sadness': (-0.5, -0.4),             
        'depression': (-0.6, -0.3),        
        'disappointment': (-0.4, -0.2),
        'grief': (-0.7, -0.1),
        'nervousness': (-0.2, 0.5),         
        'fear': (-0.5, 0.3),                
        'anger': (-0.5, 0.6),               
        'rage': (-0.7, 0.8),
        'hate': (-0.7, 0.4),           
        'disgust': (-0.6, 0.2),             
        'anxiety': (-0.4, 0.5),            
        'loneliness': (-0.5, -0.2),         
        'frustration': (-0.4, 0.4),        
        
        # Social emotions
        'shame': (-0.5, 0.1),               
        'sympathy': (0.3, 0.1),
        'affection': (0.6, 0.2),
        'attraction': (0.5, 0.3),
        'trust': (0.4, -0.1),
        }
        for term, coords in empath_fallbacks.items():
            if term not in mapping:
                mapping[term] = coords
    
        print(f"Empath circumplex mapping created with {len(mapping)} terms")
        if hasattr(self, 'vad_lexicon') and self.vad_lexicon:
            nrc_count = len([k for k in mapping.keys() if k in self.vad_lexicon])
            print(f"  NRC-VAD based: {nrc_count}")
            print(f"  Manual fallbacks: {len(mapping) - nrc_count}")
    
        return mapping

    def _get_coords(self, category_name):
        """Get circumplex coordinates for an empath category"""
        if category_name in self.cache:
            return self.cache[category_name]

        # First try direct mapping
        if category_name in self.empath_circumplex_map:
            coords = self.empath_circumplex_map[category_name]
            self.cache[category_name] = coords
            return coords

        # Clean name for fallback
        clean_name = category_name.replace('empath_', '').replace('_', ' ').lower()
        tokens = clean_name.split()

        # Try NRC-VAD first
        vals = [self.vad_lexicon[t] for t in tokens if t in self.vad_lexicon]
        if vals:
            val = sum(v['valence'] for v in vals) / len(vals)
            aro = sum(v['arousal'] for v in vals) / len(vals)
        else:
            # Fallback to VADER
            scores = self.vader.polarity_scores(clean_name)
            val = scores['compound']
            aro = max(scores['pos'], scores['neg']) * 2 - 0.5

        val = max(-1.0, min(1.0, val))
        aro = max(-1.0, min(1.0, aro))
        self.cache[category_name] = (val, aro)
        return (val, aro)
    
    def map_post(self, feature_dict, min_score=0.001):
        """
        Map a post's empath features to (valence, arousal)
        using a weighted average approach.
        """
        val_total = 0.0
        aro_total = 0.0
        total_weight = 0.0

        for feat, score in feature_dict.items():
            if score > min_score:
                val, aro = self._get_coords(feat)
                val_total += val * score
                aro_total += aro * score
                total_weight += score

        if total_weight == 0:
            return (0.0, 0.0)

        return (val_total / total_weight, aro_total / total_weight)

    def map_dataframe(self, df, prefix='empath_', id_col=None):
        """Map an entire dataframe of Empath features."""
        empath_cols = [col for col in df.columns if col.startswith(prefix)]
        
        valences = []
        arousals = []
        
        for _, row in df.iterrows():
            feature_dict = {col: row[col] for col in empath_cols if row[col] > 0}
            valence, arousal = self.map_post(feature_dict)
            valences.append(valence)
            arousals.append(arousal)
        
        result_df = df.copy()
        result_df['valence'] = valences
        result_df['arousal'] = arousals
        
        return result_df

    def summarize_coverage(self):
        """Summarize mapping coverage statistics."""
        return {
            'expert_mappings': len(self.empath_circumplex_map),
            'nrc_vad_terms': len(self.vad_lexicon),
            'cached_vader': len(self.cache) if hasattr(self, 'cache') else 0
        }
        
    def map_single_category(self, category_name):
        """Map a single category name to valence/arousal coordinates."""
        return self._get_coords(f'empath_{category_name}')

    def get_mapping_coverage(self, empath_features):
        """Analyze how features are being mapped (expert vs fallback)."""
        expert_mapped = 0
        fallback_mapped = 0
        
        for feature in empath_features:
            clean_name = feature.replace('empath_', '').replace('_', ' ').lower()
            if clean_name in self.empath_circumplex_map:
                expert_mapped += 1
            else:
                fallback_mapped += 1
        
        return {
            'total_features': len(empath_features),
            'expert_mapped': expert_mapped,
            'fallback_mapped': fallback_mapped,
            'expert_coverage': expert_mapped / len(empath_features) * 100 if empath_features else 0,
            'fallback_coverage': fallback_mapped / len(empath_features) * 100 if empath_features else 0
        }