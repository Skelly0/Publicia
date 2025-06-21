"""
Comprehensive test suite to compare reranking enabled vs disabled performance.
Tests both simple and complex queries to evaluate the effectiveness of the adaptive reranking system.
"""
import asyncio
import logging
import sys
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
import statistics

# Add the parent directory to the path so we can import from the project
sys.path.insert(0, str(Path(__file__).parent.parent))

from managers.config import Config
from managers.documents import DocumentManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RerankingTestSuite:
    """Test suite for comparing reranking enabled vs disabled performance."""
    
    def __init__(self):
        self.config = Config()
        self.doc_manager = None
        self.test_queries = {
            'simple': [
                "what is mong",
                "mong definition",
                "arshtini meaning",
                "mong culture",
                "mong people"
            ],
            'complex': [
                "Write a detailed analysis of Mong philosophy as well as its intersection with their theology",
                "Provide a comprehensive examination of the relationship between Mong cultural practices and their religious beliefs",
                "Analyze the philosophical foundations of Mong society and how they relate to their theological framework",
                "Give me a detailed overview of Mong traditions and their connection to spiritual practices",
                "Explain in detail the intersection of Mong philosophy, theology, and cultural identity",
                "Describe the comprehensive relationship between Mong belief systems and their practical applications",
                "Write about the detailed analysis of Mong religious practices and their philosophical underpinnings"
            ]
        }
        
    async def setup(self):
        """Initialize the document manager and load documents."""
        logger.info("Setting up test environment...")
        self.doc_manager = DocumentManager(config=self.config)
        await self.doc_manager._load_documents()
        logger.info(f"Loaded {len(self.doc_manager.metadata)} documents")
        
    async def run_search_test(self, query: str, apply_reranking: bool, top_k: int = 10) -> Dict[str, Any]:
        """Run a single search test and return detailed results."""
        start_time = time.time()
        
        try:
            results = await self.doc_manager.search(query, top_k=top_k, apply_reranking=apply_reranking)
            end_time = time.time()
            
            return {
                'query': query,
                'reranking_enabled': apply_reranking,
                'num_results': len(results),
                'execution_time': end_time - start_time,
                'results': results,
                'success': True,
                'error': None
            }
        except Exception as e:
            end_time = time.time()
            logger.error(f"Error in search test for query '{query}' (reranking={apply_reranking}): {e}")
            return {
                'query': query,
                'reranking_enabled': apply_reranking,
                'num_results': 0,
                'execution_time': end_time - start_time,
                'results': [],
                'success': False,
                'error': str(e)
            }
    
    def analyze_results(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze test results and provide comprehensive metrics."""
        reranking_enabled_results = [r for r in test_results if r['reranking_enabled'] and r['success']]
        reranking_disabled_results = [r for r in test_results if not r['reranking_enabled'] and r['success']]
        
        def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not results:
                return {'count': 0, 'avg_results': 0, 'avg_time': 0, 'zero_result_queries': 0}
                
            result_counts = [r['num_results'] for r in results]
            execution_times = [r['execution_time'] for r in results]
            zero_result_queries = len([r for r in results if r['num_results'] == 0])
            
            return {
                'count': len(results),
                'avg_results': statistics.mean(result_counts) if result_counts else 0,
                'median_results': statistics.median(result_counts) if result_counts else 0,
                'min_results': min(result_counts) if result_counts else 0,
                'max_results': max(result_counts) if result_counts else 0,
                'avg_time': statistics.mean(execution_times) if execution_times else 0,
                'median_time': statistics.median(execution_times) if execution_times else 0,
                'zero_result_queries': zero_result_queries,
                'zero_result_percentage': (zero_result_queries / len(results) * 100) if results else 0
            }
        
        enabled_metrics = calculate_metrics(reranking_enabled_results)
        disabled_metrics = calculate_metrics(reranking_disabled_results)
        
        # Calculate improvements
        result_improvement = 0
        time_overhead = 0
        zero_result_improvement = 0
        
        if disabled_metrics['avg_results'] > 0:
            result_improvement = ((enabled_metrics['avg_results'] - disabled_metrics['avg_results']) / 
                                disabled_metrics['avg_results'] * 100)
        
        if disabled_metrics['avg_time'] > 0:
            time_overhead = ((enabled_metrics['avg_time'] - disabled_metrics['avg_time']) / 
                           disabled_metrics['avg_time'] * 100)
        
        zero_result_improvement = disabled_metrics['zero_result_percentage'] - enabled_metrics['zero_result_percentage']
        
        return {
            'reranking_enabled': enabled_metrics,
            'reranking_disabled': disabled_metrics,
            'improvements': {
                'result_count_improvement_percent': result_improvement,
                'time_overhead_percent': time_overhead,
                'zero_result_reduction_percent': zero_result_improvement
            }
        }
    
    def print_detailed_results(self, test_results: List[Dict[str, Any]], query_type: str):
        """Print detailed results for a specific query type."""
        print(f"\n{'='*60}")
        print(f"DETAILED RESULTS FOR {query_type.upper()} QUERIES")
        print(f"{'='*60}")
        
        # Group results by query
        queries = list(set(r['query'] for r in test_results))
        
        for query in queries:
            query_results = [r for r in test_results if r['query'] == query]
            enabled_result = next((r for r in query_results if r['reranking_enabled']), None)
            disabled_result = next((r for r in query_results if not r['reranking_enabled']), None)
            
            print(f"\nQuery: {query}")
            print(f"{'-' * 80}")
            
            if enabled_result:
                print(f"WITH RERANKING:")
                print(f"  Results: {enabled_result['num_results']}")
                print(f"  Time: {enabled_result['execution_time']:.3f}s")
                if enabled_result['results']:
                    print(f"  Top result score: {enabled_result['results'][0][3]:.4f}")
                    print(f"  Top result source: {enabled_result['results'][0][1]}")
                    print(f"  Top result preview: {enabled_result['results'][0][2][:100]}...")
            
            if disabled_result:
                print(f"WITHOUT RERANKING:")
                print(f"  Results: {disabled_result['num_results']}")
                print(f"  Time: {disabled_result['execution_time']:.3f}s")
                if disabled_result['results']:
                    print(f"  Top result score: {disabled_result['results'][0][3]:.4f}")
                    print(f"  Top result source: {disabled_result['results'][0][1]}")
                    print(f"  Top result preview: {disabled_result['results'][0][2][:100]}...")
            
            if enabled_result and disabled_result:
                result_diff = enabled_result['num_results'] - disabled_result['num_results']
                time_diff = enabled_result['execution_time'] - disabled_result['execution_time']
                print(f"COMPARISON:")
                print(f"  Result difference: {result_diff:+d}")
                print(f"  Time difference: {time_diff:+.3f}s")
    
    async def run_comprehensive_test(self):
        """Run comprehensive tests comparing reranking enabled vs disabled."""
        print("="*80)
        print("COMPREHENSIVE RERANKING COMPARISON TEST")
        print("="*80)
        
        await self.setup()
        
        all_test_results = []
        
        # Test both simple and complex queries
        for query_type, queries in self.test_queries.items():
            print(f"\nTesting {query_type.upper()} queries...")
            
            query_results = []
            
            for query in queries:
                print(f"  Testing: {query[:50]}{'...' if len(query) > 50 else ''}")
                
                # Test with reranking enabled
                result_enabled = await self.run_search_test(query, apply_reranking=True)
                query_results.append(result_enabled)
                all_test_results.append(result_enabled)
                
                # Test with reranking disabled
                result_disabled = await self.run_search_test(query, apply_reranking=False)
                query_results.append(result_disabled)
                all_test_results.append(result_disabled)
                
                # Brief pause between queries
                await asyncio.sleep(0.1)
            
            # Analyze results for this query type
            analysis = self.analyze_results(query_results)
            
            print(f"\n{query_type.upper()} QUERIES SUMMARY:")
            print(f"{'-' * 40}")
            print(f"Reranking ENABLED  - Avg Results: {analysis['reranking_enabled']['avg_results']:.1f}, "
                  f"Avg Time: {analysis['reranking_enabled']['avg_time']:.3f}s, "
                  f"Zero Results: {analysis['reranking_enabled']['zero_result_percentage']:.1f}%")
            print(f"Reranking DISABLED - Avg Results: {analysis['reranking_disabled']['avg_results']:.1f}, "
                  f"Avg Time: {analysis['reranking_disabled']['avg_time']:.3f}s, "
                  f"Zero Results: {analysis['reranking_disabled']['zero_result_percentage']:.1f}%")
            print(f"IMPROVEMENTS:")
            print(f"  Result Count: {analysis['improvements']['result_count_improvement_percent']:+.1f}%")
            print(f"  Time Overhead: {analysis['improvements']['time_overhead_percent']:+.1f}%")
            print(f"  Zero Result Reduction: {analysis['improvements']['zero_result_reduction_percent']:+.1f}%")
            
            # Print detailed results
            self.print_detailed_results(query_results, query_type)
        
        # Overall analysis
        overall_analysis = self.analyze_results(all_test_results)
        
        print(f"\n{'='*80}")
        print("OVERALL COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"Total queries tested: {len(self.test_queries['simple']) + len(self.test_queries['complex'])}")
        print(f"Total test runs: {len(all_test_results)}")
        
        print(f"\nOVERALL METRICS:")
        print(f"{'-' * 40}")
        print(f"Reranking ENABLED:")
        print(f"  Average results per query: {overall_analysis['reranking_enabled']['avg_results']:.2f}")
        print(f"  Average execution time: {overall_analysis['reranking_enabled']['avg_time']:.3f}s")
        print(f"  Queries with zero results: {overall_analysis['reranking_enabled']['zero_result_percentage']:.1f}%")
        
        print(f"\nReranking DISABLED:")
        print(f"  Average results per query: {overall_analysis['reranking_disabled']['avg_results']:.2f}")
        print(f"  Average execution time: {overall_analysis['reranking_disabled']['avg_time']:.3f}s")
        print(f"  Queries with zero results: {overall_analysis['reranking_disabled']['zero_result_percentage']:.1f}%")
        
        print(f"\nOVERALL IMPROVEMENTS WITH RERANKING:")
        print(f"{'-' * 40}")
        print(f"Result count improvement: {overall_analysis['improvements']['result_count_improvement_percent']:+.1f}%")
        print(f"Time overhead: {overall_analysis['improvements']['time_overhead_percent']:+.1f}%")
        print(f"Zero result reduction: {overall_analysis['improvements']['zero_result_reduction_percent']:+.1f}%")
        
        # Recommendations
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS")
        print(f"{'='*80}")
        
        if overall_analysis['improvements']['result_count_improvement_percent'] > 10:
            print("[GOOD] RERANKING ENABLED shows significant improvement in result count")
        elif overall_analysis['improvements']['result_count_improvement_percent'] > 0:
            print("[GOOD] RERANKING ENABLED shows modest improvement in result count")
        else:
            print("[POOR] RERANKING ENABLED does not improve result count")
        
        if overall_analysis['improvements']['zero_result_reduction_percent'] > 10:
            print("[GOOD] RERANKING ENABLED significantly reduces queries with zero results")
        elif overall_analysis['improvements']['zero_result_reduction_percent'] > 0:
            print("[GOOD] RERANKING ENABLED reduces queries with zero results")
        else:
            print("[POOR] RERANKING ENABLED does not reduce zero result queries")
        
        if overall_analysis['improvements']['time_overhead_percent'] < 50:
            print("[GOOD] RERANKING ENABLED has acceptable time overhead")
        else:
            print("[WARNING] RERANKING ENABLED has high time overhead")
        
        # Final recommendation
        result_improvement = overall_analysis['improvements']['result_count_improvement_percent']
        zero_result_improvement = overall_analysis['improvements']['zero_result_reduction_percent']
        time_overhead = overall_analysis['improvements']['time_overhead_percent']
        
        if result_improvement > 5 or zero_result_improvement > 5:
            if time_overhead < 100:  # Less than 2x time
                print(f"\n[RECOMMENDATION] ENABLE RERANKING")
                print("   Benefits outweigh the time cost")
            else:
                print(f"\n[RECOMMENDATION] CONSIDER RERANKING")
                print("   Good results but high time cost - evaluate based on use case")
        else:
            print(f"\n[RECOMMENDATION] DISABLE RERANKING")
            print("   Minimal benefits do not justify the overhead")
        
        return overall_analysis

async def main():
    """Main function to run the reranking comparison test."""
    test_suite = RerankingTestSuite()
    
    try:
        analysis = await test_suite.run_comprehensive_test()
        return analysis
    except Exception as e:
        logger.error(f"Error running test suite: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    asyncio.run(main())