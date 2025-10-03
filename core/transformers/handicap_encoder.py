#!/usr/bin/env python3
"""
French Handicap Classification Encoder

This module provides functions to parse and encode French handicap classifications
from text into structured numeric features for machine learning models.

French Handicap System Hierarchy:
1. "Handicap de Catégorie" (Category Handicap) - Higher prestige
2. "Handicap" (Standard Handicap) - Lower prestige  
3. Non-handicap races - Lowest

Division Hierarchy (within handicaps):
- "première épreuve" (1st division) - Highest class within handicap type
- "deuxième épreuve" (2nd division) 
- "troisième épreuve" (3rd division)
- "quatrième épreuve" (4th division) - Lowest class
- Undivided handicaps rank between 1st and 2nd divisions
"""

import re
from typing import Dict, Tuple, Optional


class HandicapEncoder:
    """Encoder for French handicap classifications."""
    
    # Division keywords and their numeric values (1 = highest, 4 = lowest)
    DIVISION_KEYWORDS = {
        'première': 1,
        'deuxième': 2,
        'troisième': 3,  
        'quatrième': 4
    }
    
    # Base scores for handicap types
    CATEGORY_HANDICAP_BASE = 1000  # "Handicap de Catégorie"
    STANDARD_HANDICAP_BASE = 500   # "Handicap" 
    NON_HANDICAP_BASE = 0          # Not a handicap race
    
    # Division score modifiers (higher number = higher prestige)
    DIVISION_SCORES = {
        0: 150,  # Undivided (between 1st and 2nd)
        1: 200,  # première (highest division)
        2: 100,  # deuxième  
        3: 50,   # troisième
        4: 25    # quatrième (lowest division)
    }
    
    @staticmethod
    def parse_handicap_text(handicap_text: str) -> Dict[str, any]:
        """
        Parse French handicap text into structured components.
        
        Args:
            handicap_text: Raw handicap text from MySQL (e.g., "Handicap de Catégorie divisé - première épreuve")
            
        Returns:
            Dictionary with parsed handicap components:
            {
                'handi_raw': str,              # Original text
                'is_handicap': bool,           # True if any type of handicap
                'is_category_handicap': bool,  # True if "Handicap de Catégorie"
                'handicap_division': int,      # 0=undivided, 1-4=division number
                'handicap_level_score': int    # Numeric hierarchy score
            }
        """
        # Clean input
        if not handicap_text or not isinstance(handicap_text, str):
            handicap_text = ""
        
        original_text = handicap_text.strip()
        text = original_text.lower()
        
        # Initialize result
        result = {
            'handi_raw': original_text,
            'is_handicap': False,
            'is_category_handicap': False, 
            'handicap_division': 0,
            'handicap_level_score': HandicapEncoder.NON_HANDICAP_BASE
        }
        
        # Check if it's a handicap at all
        if 'handicap' not in text:
            return result
            
        result['is_handicap'] = True
        
        # Determine handicap type
        if 'handicap de catégorie' in text or 'handicap de categorie' in text:
            result['is_category_handicap'] = True
            base_score = HandicapEncoder.CATEGORY_HANDICAP_BASE
        else:
            base_score = HandicapEncoder.STANDARD_HANDICAP_BASE
            
        # Determine division
        division = 0  # Default to undivided
        
        # Check for division keywords
        for division_name, division_num in HandicapEncoder.DIVISION_KEYWORDS.items():
            if division_name in text:
                division = division_num
                break
        
        result['handicap_division'] = division
        
        # Calculate final score
        division_score = HandicapEncoder.DIVISION_SCORES.get(division, 0)
        result['handicap_level_score'] = base_score + division_score
        
        return result
    
    @staticmethod
    def get_handicap_prestige_rank(handicap_level_score: int) -> str:
        """
        Get human-readable prestige ranking from level score.
        
        Args:
            handicap_level_score: Numeric handicap level score
            
        Returns:
            String describing prestige level
        """
        if handicap_level_score >= 1100:
            return "Category Handicap - Premier Division"
        elif handicap_level_score >= 1050:
            return "Category Handicap - Undivided" 
        elif handicap_level_score >= 1025:
            return "Category Handicap - Lower Division"
        elif handicap_level_score >= 700:
            return "Standard Handicap - Premier Division"
        elif handicap_level_score >= 650:
            return "Standard Handicap - Undivided"
        elif handicap_level_score >= 525:
            return "Standard Handicap - Lower Division"
        else:
            return "Non-Handicap Race"
    
    @staticmethod
    def encode_batch(handicap_texts: list) -> list:
        """
        Encode a batch of handicap texts.
        
        Args:
            handicap_texts: List of handicap text strings
            
        Returns:
            List of encoded handicap dictionaries
        """
        return [HandicapEncoder.parse_handicap_text(text) for text in handicap_texts]


def test_handicap_encoder():
    """Test function to validate the handicap encoder."""
    
    test_cases = [
        "",
        "Handicap",
        "Handicap divisé - première épreuve", 
        "Handicap divisé - deuxième épreuve",
        "Handicap divisé - troisième épreuve",
        "Handicap divisé - quatrième épreuve",
        "Handicap de Catégorie", 
        "Handicap de Catégorie divisé - première épreuve",
        "Handicap de Catégorie divisé - deuxième épreuve",
        "Handicap de Catégorie divisé - troisième épreuve", 
        "Handicap de Catégorie divisé - quatrième épreuve",
        "Course normale",  # Non-handicap
    ]
    
    print("=== HANDICAP ENCODER TEST ===\n")
    
    results = []
    for text in test_cases:
        encoded = HandicapEncoder.parse_handicap_text(text)
        results.append((text, encoded))
        
        print(f"Input: \"{text}\"")
        print(f"  - Is Handicap: {encoded['is_handicap']}")
        print(f"  - Category Handicap: {encoded['is_category_handicap']}")
        print(f"  - Division: {encoded['handicap_division']}")
        print(f"  - Level Score: {encoded['handicap_level_score']}")
        print(f"  - Prestige: {HandicapEncoder.get_handicap_prestige_rank(encoded['handicap_level_score'])}")
        print()
    
    # Verify hierarchy is correct (higher scores = higher prestige)
    print("=== HIERARCHY VALIDATION ===\n")
    
    # Sort by level score descending
    sorted_results = sorted(results, key=lambda x: x[1]['handicap_level_score'], reverse=True)
    
    print("Handicap hierarchy (highest to lowest prestige):")
    for i, (text, encoded) in enumerate(sorted_results, 1):
        if encoded['is_handicap']:  # Only show handicap races
            print(f"{i:2d}. {encoded['handicap_level_score']:4d} - \"{text}\"")
    
    return results


if __name__ == "__main__":
    test_handicap_encoder()