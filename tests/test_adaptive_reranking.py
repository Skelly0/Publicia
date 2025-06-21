"""
Test suite specifically for adaptive reranking settings detection.
Tests whether the system correctly identifies complex vs simple queries and applies appropriate settings.
"""
import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import from the project
sys.path.insert(0, str(Path(__file__).parent.parent))

from managers.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptiveRerankingTest:
    """Test suite for adaptive reranking settings detection."""
    
    def __init__(self):
        self.config = Config()
        
    def test_query_complexity_detection(self):
        """Test that queries are correctly classified as simple or complex."""
        
        simple_queries = [
            "what is mong",
            "mong definition",
            "arshtini meaning",
            "mong culture",
            "mong people",
            "who are the mong",
            "mong history",
            "basic facts about mong"
        ]
        
        complex_queries = [
            "Write a detailed analysis of Mong philosophy as well as its intersection with their theology",
            "Provide a comprehensive examination of the relationship between Mong cultural practices",
            "Analyze the philosophical foundations of Mong society",
            "Give me a detailed overview of Mong traditions",
            "Explain in detail the intersection of Mong philosophy",
            "Describe the comprehensive relationship between Mong belief systems",
            "Write about the detailed analysis of Mong religious practices",
            "Tell me about the complex relationship between Mong culture and religion",
            "Provide an analysis of Mong theological concepts",
            "Discuss the elaborate traditions of the Mong people",
            "Research the deep connections between Mong philosophy and practice",
            "Investigate the comprehensive nature of Mong belief systems"
        ]
        
        print("="*80)
        print("ADAPTIVE RERANKING SETTINGS TEST")
        print("="*80)
        
        print("\nTesting SIMPLE queries (should use standard settings):")
        print("-" * 60)
        
        simple_correct = 0
        for query in simple_queries:
            settings = self.config.get_reranking_settings_for_query(query)
            is_detected_as_simple = settings['filter_mode'] != 'topk'
            
            print(f"Query: {query}")
            print(f"  Settings: {settings}")
            print(f"  Detected as: {'SIMPLE' if is_detected_as_simple else 'COMPLEX'}")
            print(f"  Correct: {'YES' if is_detected_as_simple else 'NO'}")
            print()
            
            if is_detected_as_simple:
                simple_correct += 1
        
        print(f"Simple query detection accuracy: {simple_correct}/{len(simple_queries)} ({simple_correct/len(simple_queries)*100:.1f}%)")
        
        print("\nTesting COMPLEX queries (should use lenient settings):")
        print("-" * 60)
        
        complex_correct = 0
        for query in complex_queries:
            settings = self.config.get_reranking_settings_for_query(query)
            is_detected_as_complex = settings['filter_mode'] == 'topk'
            
            print(f"Query: {query[:60]}{'...' if len(query) > 60 else ''}")
            print(f"  Settings: {settings}")
            print(f"  Detected as: {'COMPLEX' if is_detected_as_complex else 'SIMPLE'}")
            print(f"  Correct: {'YES' if is_detected_as_complex else 'NO'}")
            print()
            
            if is_detected_as_complex:
                complex_correct += 1
        
        print(f"Complex query detection accuracy: {complex_correct}/{len(complex_queries)} ({complex_correct/len(complex_queries)*100:.1f}%)")
        
        # Overall accuracy
        total_correct = simple_correct + complex_correct
        total_queries = len(simple_queries) + len(complex_queries)
        overall_accuracy = total_correct / total_queries * 100
        
        print(f"\n{'='*80}")
        print("OVERALL RESULTS")
        print(f"{'='*80}")
        print(f"Total queries tested: {total_queries}")
        print(f"Correctly classified: {total_correct}")
        print(f"Overall accuracy: {overall_accuracy:.1f}%")
        
        if overall_accuracy >= 90:
            print("[EXCELLENT] Adaptive reranking detection is working very well")
        elif overall_accuracy >= 80:
            print("[GOOD] Adaptive reranking detection is working well")
        elif overall_accuracy >= 70:
            print("[FAIR] Adaptive reranking detection needs improvement")
        else:
            print("[POOR] Adaptive reranking detection needs significant improvement")
        
        return {
            'simple_accuracy': simple_correct / len(simple_queries) * 100,
            'complex_accuracy': complex_correct / len(complex_queries) * 100,
            'overall_accuracy': overall_accuracy,
            'total_queries': total_queries,
            'correct_classifications': total_correct
        }
    
    def test_settings_differences(self):
        """Test that different settings are actually applied for simple vs complex queries."""
        
        simple_query = "what is mong"
        complex_query = "Write a detailed analysis of Mong philosophy as well as its intersection with their theology"
        
        simple_settings = self.config.get_reranking_settings_for_query(simple_query)
        complex_settings = self.config.get_reranking_settings_for_query(complex_query)
        
        print(f"\n{'='*80}")
        print("SETTINGS COMPARISON")
        print(f"{'='*80}")
        
        print(f"Simple query: '{simple_query}'")
        print(f"Settings: {simple_settings}")
        print()
        
        print(f"Complex query: '{complex_query[:60]}...'")
        print(f"Settings: {complex_settings}")
        print()
        
        print("DIFFERENCES:")
        print("-" * 40)
        
        differences = []
        
        for key in simple_settings:
            if simple_settings[key] != complex_settings[key]:
                print(f"  {key}: {simple_settings[key]} -> {complex_settings[key]}")
                differences.append(key)
            else:
                print(f"  {key}: {simple_settings[key]} (same)")
        
        print(f"\nNumber of different settings: {len(differences)}")
        
        if len(differences) >= 2:
            print("[GOOD] Settings are appropriately different between simple and complex queries")
        elif len(differences) == 1:
            print("[FAIR] Some settings differ, but could be more adaptive")
        else:
            print("[POOR] Settings are the same for both query types")
        
        return {
            'simple_settings': simple_settings,
            'complex_settings': complex_settings,
            'differences': differences,
            'num_differences': len(differences)
        }
    
    def test_keyword_coverage(self):
        """Test the coverage of complexity indicators."""
        
        print(f"\n{'='*80}")
        print("KEYWORD COVERAGE TEST")
        print(f"{'='*80}")
        
        # Test individual keywords
        complexity_keywords = [
            'analysis', 'analyze', 'detailed', 'comprehensive', 'intersection',
            'relationship', 'compare', 'contrast', 'philosophy', 'theology',
            'write about', 'explain in detail', 'discuss', 'elaborate',
            'write a', 'provide a', 'give me a', 'tell me',
            'describe', 'overview', 'summary', 'breakdown', 'examination',
            'exploration', 'investigation', 'study', 'research', 'deep dive'
        ]
        
        print("Testing individual complexity keywords:")
        print("-" * 50)
        
        detected_keywords = 0
        for keyword in complexity_keywords:
            test_query = f"Please {keyword} the mong culture"
            settings = self.config.get_reranking_settings_for_query(test_query)
            is_complex = settings['filter_mode'] == 'topk'
            
            print(f"  '{keyword}': {'YES' if is_complex else 'NO'}")
            
            if is_complex:
                detected_keywords += 1
        
        keyword_coverage = detected_keywords / len(complexity_keywords) * 100
        print(f"\nKeyword coverage: {detected_keywords}/{len(complexity_keywords)} ({keyword_coverage:.1f}%)")
        
        if keyword_coverage >= 90:
            print("[EXCELLENT] Keyword coverage is comprehensive")
        elif keyword_coverage >= 80:
            print("[GOOD] Keyword coverage is good")
        elif keyword_coverage >= 70:
            print("[FAIR] Keyword coverage could be improved")
        else:
            print("[POOR] Keyword coverage is insufficient")
        
        return {
            'total_keywords': len(complexity_keywords),
            'detected_keywords': detected_keywords,
            'coverage_percentage': keyword_coverage
        }
    
    def run_all_tests(self):
        """Run all adaptive reranking tests."""
        
        print("Starting Adaptive Reranking Test Suite...")
        print("This tests whether the system correctly detects query complexity")
        print("and applies appropriate reranking settings.\n")
        
        # Test 1: Query complexity detection
        detection_results = self.test_query_complexity_detection()
        
        # Test 2: Settings differences
        settings_results = self.test_settings_differences()
        
        # Test 3: Keyword coverage
        keyword_results = self.test_keyword_coverage()
        
        # Summary
        print(f"\n{'='*80}")
        print("TEST SUITE SUMMARY")
        print(f"{'='*80}")
        
        print(f"Query Classification Accuracy: {detection_results['overall_accuracy']:.1f}%")
        print(f"Settings Differentiation: {settings_results['num_differences']} parameters differ")
        print(f"Keyword Coverage: {keyword_results['coverage_percentage']:.1f}%")
        
        # Overall assessment
        overall_score = (
            detection_results['overall_accuracy'] * 0.5 +
            (settings_results['num_differences'] / 3 * 100) * 0.3 +
            keyword_results['coverage_percentage'] * 0.2
        )
        
        print(f"\nOverall Adaptive Reranking Score: {overall_score:.1f}/100")
        
        if overall_score >= 85:
            print("[EXCELLENT] Adaptive reranking system is working very well")
        elif overall_score >= 75:
            print("[GOOD] Adaptive reranking system is working well")
        elif overall_score >= 65:
            print("[FAIR] Adaptive reranking system needs some improvements")
        else:
            print("[POOR] Adaptive reranking system needs significant improvements")
        
        return {
            'detection_results': detection_results,
            'settings_results': settings_results,
            'keyword_results': keyword_results,
            'overall_score': overall_score
        }

def main():
    """Main function to run the adaptive reranking test."""
    test_suite = AdaptiveRerankingTest()
    
    try:
        results = test_suite.run_all_tests()
        return results
    except Exception as e:
        logger.error(f"Error running adaptive reranking test: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    main()