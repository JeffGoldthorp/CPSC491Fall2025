"""
Compare fine-tuned model responses (from prompts2.txt) with base GPT-4o-mini model responses.
"""

import openai
from openai import OpenAI
import re
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from config import get_api_key

# Initialize OpenAI client
client = OpenAI(api_key=get_api_key())

def extract_questions_and_responses(filepath: str) -> List[Tuple[str, str]]:
    """Extract questions and fine-tuned model responses from prompts2.txt."""
    qa_pairs = []
    
    # Try different encodings
    for encoding in ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1']:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
            if content.strip():
                print(f"  ✓ Successfully read file with {encoding} encoding")
                break
        except:
            continue
    
    # Split by the question marker - try different variations
    if '🧑‍💻 You:' in content:
        sections = content.split('🧑‍💻 You:')
    elif 'You:' in content:
        sections = content.split('You:')
    else:
        print("  ❌ Could not find question marker")
        return []
    
    print(f"  Found {len(sections)} sections")
    
    for idx, section in enumerate(sections[1:], 1):  # Skip first empty section
        print(f"\n  Processing section {idx}...")
        lines = section.strip().split('\n')
        if not lines:
            print("    ❌ No lines found")
            continue
        
        # First line is the question
        question = lines[0].strip()
        print(f"    Question: {question[:80]}...")
        if not question:
            print("    ❌ Empty question")
            continue
        
        # Find "Assistant:" to get the response
        assistant_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('Assistant:'):
                assistant_idx = i
                print(f"    Found Assistant at line {i}")
                break
        
        if assistant_idx == -1:
            print("    ❌ No Assistant: found")
            continue
        
        # Collect all lines from Assistant: until we hit "📚 Sources:" or next question
        response_lines = []
        for i in range(assistant_idx, len(lines)):
            line = lines[i]
            if '📚 Sources:' in line or '🧑‍💻 You:' in line:
                break
            # Skip the "Assistant:" prefix on first line
            if i == assistant_idx:
                line = line.replace('Assistant:', '').strip()
            response_lines.append(line)
        
        response = '\n'.join(response_lines).strip()
        print(f"    Response length: {len(response)} chars")
        
        if question and response and len(response) > 50:  # Valid response
            qa_pairs.append((question, response))
            print(f"    ✓ Extracted Q{len(qa_pairs)}")
        else:
            print(f"    ❌ Invalid (response too short: {len(response)} chars)")
    
    return qa_pairs

def get_base_model_response(question: str, model: str = "gpt-4o-mini") -> str:
    """Get response from base GPT-4o-mini model."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant knowledgeable about emergency alerting systems, including EAS, WEA, IPAWS, and related topics."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error getting response: {e}")
        return ""

def get_embedding(text: str) -> np.ndarray:
    """Get OpenAI embedding for text."""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"❌ Error getting embedding: {e}")
        return np.zeros(1536)

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts."""
    emb1 = get_embedding(text1).reshape(1, -1)
    emb2 = get_embedding(text2).reshape(1, -1)
    return cosine_similarity(emb1, emb2)[0][0]

def analyze_relevance_to_emergency_systems(response: str) -> float:
    """Calculate how relevant a response is to emergency alerting systems."""
    reference_topics = [
        "Emergency Alert System EAS broadcast standards public safety",
        "Wireless Emergency Alerts WEA mobile notifications IPAWS",
        "Federal Communications Commission FCC emergency alerting regulations",
        "FEMA disaster alerts emergency management coordination",
        "Public warning systems reliability security accessibility",
        "Emergency notification technology broadcasting infrastructure",
        "Alert message design delivery timing public response"
    ]
    
    response_emb = get_embedding(response).reshape(1, -1)
    
    similarities = []
    for topic in reference_topics:
        topic_emb = get_embedding(topic).reshape(1, -1)
        sim = cosine_similarity(response_emb, topic_emb)[0][0]
        similarities.append(sim)
    
    return max(similarities)

def check_for_sources(response: str) -> bool:
    """Check if response cites sources or provides references."""
    source_indicators = [
        'source:', 'sources:', 'according to', 'reported by', 'study by',
        'research from', 'published in', '[', 'http://', 'https://',
        'reference:', 'citation:', 'see:', 'cf.', 'ibid'
    ]
    response_lower = response.lower()
    return any(indicator in response_lower for indicator in source_indicators)

def check_for_hedging(response: str) -> int:
    """Count hedging/uncertainty phrases that might indicate accuracy concerns."""
    hedging_phrases = [
        'may', 'might', 'could', 'possibly', 'perhaps', 'likely',
        'probably', 'generally', 'typically', 'often', 'usually',
        'appears to', 'seems to', 'suggests that', 'indicates that',
        'it is possible', 'it is likely'
    ]
    response_lower = response.lower()
    count = sum(response_lower.count(phrase) for phrase in hedging_phrases)
    return count

def check_specificity(response: str) -> Dict:
    """Analyze response specificity - names, dates, numbers, etc."""
    # Count specific entities
    import re
    
    # Count years/dates (4-digit years)
    years = len(re.findall(r'\b(19|20)\d{2}\b', response))
    
    # Count numbers/statistics
    numbers = len(re.findall(r'\b\d+\.?\d*%?\b', response))
    
    # Count proper nouns (capitalized words, excluding sentence starts)
    sentences = response.split('.')
    proper_nouns = 0
    for sentence in sentences:
        words = sentence.split()[1:]  # Skip first word of each sentence
        proper_nouns += sum(1 for word in words if word and word[0].isupper())
    
    # Count URLs and citations
    urls = len(re.findall(r'https?://', response))
    
    return {
        'years': years,
        'numbers': numbers,
        'proper_nouns': proper_nouns,
        'urls': urls,
        'specificity_score': years + numbers + proper_nouns + urls
    }

def analyze_hallucination_risk(response: str) -> Dict:
    """Analyze factors that might indicate hallucination."""
    has_sources = check_for_sources(response)
    hedging_count = check_for_hedging(response)
    specificity = check_specificity(response)
    
    # High specificity without sources = higher hallucination risk
    hallucination_risk_score = 0
    
    if specificity['specificity_score'] > 10 and not has_sources:
        hallucination_risk_score += 3  # Many specific claims without sources
    elif specificity['specificity_score'] > 5 and not has_sources:
        hallucination_risk_score += 2
    elif not has_sources:
        hallucination_risk_score += 1
    
    if hedging_count > 10:
        hallucination_risk_score -= 1  # Hedging reduces risk
    
    return {
        'has_sources': has_sources,
        'hedging_count': hedging_count,
        'specificity': specificity,
        'risk_score': max(0, hallucination_risk_score),
        'risk_level': 'Low' if hallucination_risk_score <= 1 else ('Medium' if hallucination_risk_score <= 2 else 'High')
    }

def compare_responses(question: str, finetuned_response: str, base_response: str) -> Dict:
    """Compare both responses across multiple metrics."""
    
    print(f"\n📊 Analyzing: {question[:70]}...")
    
    # Calculate relevance to emergency systems
    finetuned_relevance = analyze_relevance_to_emergency_systems(finetuned_response)
    base_relevance = analyze_relevance_to_emergency_systems(base_response)
    
    # Calculate similarity to question (directness)
    finetuned_q_sim = calculate_similarity(question, finetuned_response)
    base_q_sim = calculate_similarity(question, base_response)
    
    # Calculate response lengths
    finetuned_length = len(finetuned_response)
    base_length = len(base_response)
    
    # Calculate similarity between the two responses
    response_similarity = calculate_similarity(finetuned_response, base_response)
    
    # Analyze specificity
    finetuned_specificity = check_specificity(finetuned_response)
    base_specificity = check_specificity(base_response)
    
    # Analyze hallucination risk
    finetuned_hallucination = analyze_hallucination_risk(finetuned_response)
    base_hallucination = analyze_hallucination_risk(base_response)
    
    return {
        'question': question,
        'finetuned_response': finetuned_response,
        'base_response': base_response,
        'finetuned_relevance': finetuned_relevance,
        'base_relevance': base_relevance,
        'finetuned_q_similarity': finetuned_q_sim,
        'base_q_similarity': base_q_sim,
        'finetuned_length': finetuned_length,
        'base_length': base_length,
        'response_similarity': response_similarity,
        'finetuned_specificity': finetuned_specificity,
        'base_specificity': base_specificity,
        'finetuned_hallucination': finetuned_hallucination,
        'base_hallucination': base_hallucination
    }

def generate_comparison_report(results: List[Dict], output_file: str = "model_comparison_detailed.txt"):
    """Generate detailed comparison report."""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("MODEL COMPARISON REPORT: Fine-tuned Model vs GPT-4o-mini\n")
        f.write("="*100 + "\n\n")
        
        # Summary statistics
        avg_ft_relevance = np.mean([r['finetuned_relevance'] for r in results])
        avg_base_relevance = np.mean([r['base_relevance'] for r in results])
        avg_ft_q_sim = np.mean([r['finetuned_q_similarity'] for r in results])
        avg_base_q_sim = np.mean([r['base_q_similarity'] for r in results])
        avg_ft_length = np.mean([r['finetuned_length'] for r in results])
        avg_base_length = np.mean([r['base_length'] for r in results])
        avg_response_sim = np.mean([r['response_similarity'] for r in results])
        
        # Specificity metrics
        avg_ft_specificity = np.mean([r['finetuned_specificity']['specificity_score'] for r in results])
        avg_base_specificity = np.mean([r['base_specificity']['specificity_score'] for r in results])
        
        # Hallucination risk
        avg_ft_risk = np.mean([r['finetuned_hallucination']['risk_score'] for r in results])
        avg_base_risk = np.mean([r['base_hallucination']['risk_score'] for r in results])
        
        # Source citation
        ft_sources_count = sum(1 for r in results if r['finetuned_hallucination']['has_sources'])
        base_sources_count = sum(1 for r in results if r['base_hallucination']['has_sources'])
        
        f.write("📈 SUMMARY STATISTICS\n")
        f.write("-"*100 + "\n\n")
        
        f.write("1️⃣ RELEVANCE TO EMERGENCY ALERTING SYSTEMS (higher is better):\n")
        f.write(f"  Fine-tuned Model: {avg_ft_relevance:.4f}\n")
        f.write(f"  Base GPT-4o-mini: {avg_base_relevance:.4f}\n")
        f.write(f"  Winner: {'Fine-tuned ✅' if avg_ft_relevance > avg_base_relevance else 'Base ✅'} ")
        f.write(f"(+{abs(avg_ft_relevance - avg_base_relevance):.4f})\n\n")
        
        f.write("2️⃣ QUESTION DIRECTNESS (how directly it answers the question):\n")
        f.write(f"  Fine-tuned Model: {avg_ft_q_sim:.4f}\n")
        f.write(f"  Base GPT-4o-mini: {avg_base_q_sim:.4f}\n")
        f.write(f"  Winner: {'Fine-tuned ✅' if avg_ft_q_sim > avg_base_q_sim else 'Base ✅'} ")
        f.write(f"(+{abs(avg_ft_q_sim - avg_base_q_sim):.4f})\n\n")
        
        f.write("3️⃣ SPECIFICITY (names, dates, numbers, citations - higher indicates more detail):\n")
        f.write(f"  Fine-tuned Model: {avg_ft_specificity:.1f} specific items per response\n")
        f.write(f"  Base GPT-4o-mini: {avg_base_specificity:.1f} specific items per response\n")
        f.write(f"  Winner: {'Fine-tuned ✅' if avg_ft_specificity > avg_base_specificity else 'Base ✅'} ")
        f.write(f"(+{abs(avg_ft_specificity - avg_base_specificity):.1f})\n\n")
        
        f.write("4️⃣ SOURCE CITATION (responses with sources/references):\n")
        f.write(f"  Fine-tuned Model: {ft_sources_count}/{len(results)} ({ft_sources_count/len(results)*100:.1f}%)\n")
        f.write(f"  Base GPT-4o-mini: {base_sources_count}/{len(results)} ({base_sources_count/len(results)*100:.1f}%)\n")
        f.write(f"  Winner: {'Fine-tuned ✅' if ft_sources_count > base_sources_count else 'Base ✅'}\n\n")
        
        f.write("5️⃣ HALLUCINATION RISK (lower is better - based on specificity without sources):\n")
        f.write(f"  Fine-tuned Model: {avg_ft_risk:.2f}\n")
        f.write(f"  Base GPT-4o-mini: {avg_base_risk:.2f}\n")
        f.write(f"  Winner: {'Fine-tuned ✅' if avg_ft_risk < avg_base_risk else 'Base ✅'} ")
        f.write(f"(Δ{abs(avg_ft_risk - avg_base_risk):.2f})\n\n")
        
        f.write("6️⃣ RESPONSE LENGTH (characters):\n")
        f.write(f"  Fine-tuned Model: {avg_ft_length:.0f}\n")
        f.write(f"  Base GPT-4o-mini: {avg_base_length:.0f}\n")
        f.write(f"  Difference: {abs(avg_ft_length - avg_base_length):.0f} chars ")
        f.write(f"({'Fine-tuned more verbose' if avg_ft_length > avg_base_length else 'Base more verbose'})\n\n")
        
        f.write("7️⃣ RESPONSE SIMILARITY (how similar the responses are to each other):\n")
        f.write(f"  Average: {avg_response_sim:.4f}\n")
        f.write(f"  Interpretation: {'Very similar' if avg_response_sim > 0.8 else ('Moderately similar' if avg_response_sim > 0.6 else 'Different approaches')}\n\n")
        
        # Detailed comparisons
        f.write("\n" + "="*100 + "\n")
        f.write("📋 DETAILED QUESTION-BY-QUESTION COMPARISON\n")
        f.write("="*100 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"\n{'='*100}\n")
            f.write(f"QUESTION {i}\n")
            f.write(f"{'='*100}\n\n")
            f.write(f"❓ {result['question']}\n\n")
            
            f.write("-"*100 + "\n")
            f.write("🤖 FINE-TUNED MODEL RESPONSE:\n")
            f.write("-"*100 + "\n")
            f.write(f"{result['finetuned_response']}\n\n")
            
            f.write("-"*100 + "\n")
            f.write("💬 BASE GPT-4o-mini RESPONSE:\n")
            f.write("-"*100 + "\n")
            f.write(f"{result['base_response']}\n\n")
            
            f.write("-"*100 + "\n")
            f.write("📊 COMPARATIVE METRICS:\n")
            f.write("-"*100 + "\n\n")
            
            f.write(f"RELEVANCE TO EMERGENCY SYSTEMS:\n")
            f.write(f"  Fine-tuned: {result['finetuned_relevance']:.4f}\n")
            f.write(f"  Base:       {result['base_relevance']:.4f}\n")
            f.write(f"  Winner:     {'Fine-tuned ✅' if result['finetuned_relevance'] > result['base_relevance'] else 'Base ✅'}\n\n")
            
            f.write(f"QUESTION DIRECTNESS:\n")
            f.write(f"  Fine-tuned: {result['finetuned_q_similarity']:.4f}\n")
            f.write(f"  Base:       {result['base_q_similarity']:.4f}\n")
            f.write(f"  Winner:     {'Fine-tuned ✅' if result['finetuned_q_similarity'] > result['base_q_similarity'] else 'Base ✅'}\n\n")
            
            f.write(f"SPECIFICITY BREAKDOWN:\n")
            f.write(f"  Fine-tuned: {result['finetuned_specificity']['specificity_score']} items ")
            f.write(f"(Years: {result['finetuned_specificity']['years']}, ")
            f.write(f"Numbers: {result['finetuned_specificity']['numbers']}, ")
            f.write(f"Proper nouns: {result['finetuned_specificity']['proper_nouns']}, ")
            f.write(f"URLs: {result['finetuned_specificity']['urls']})\n")
            f.write(f"  Base:       {result['base_specificity']['specificity_score']} items ")
            f.write(f"(Years: {result['base_specificity']['years']}, ")
            f.write(f"Numbers: {result['base_specificity']['numbers']}, ")
            f.write(f"Proper nouns: {result['base_specificity']['proper_nouns']}, ")
            f.write(f"URLs: {result['base_specificity']['urls']})\n")
            f.write(f"  Winner:     {'Fine-tuned ✅' if result['finetuned_specificity']['specificity_score'] > result['base_specificity']['specificity_score'] else 'Base ✅'}\n\n")
            
            f.write(f"HALLUCINATION RISK ANALYSIS:\n")
            f.write(f"  Fine-tuned: Risk={result['finetuned_hallucination']['risk_level']} ({result['finetuned_hallucination']['risk_score']}), ")
            f.write(f"Sources={'Yes ✅' if result['finetuned_hallucination']['has_sources'] else 'No ❌'}, ")
            f.write(f"Hedging={result['finetuned_hallucination']['hedging_count']}\n")
            f.write(f"  Base:       Risk={result['base_hallucination']['risk_level']} ({result['base_hallucination']['risk_score']}), ")
            f.write(f"Sources={'Yes ✅' if result['base_hallucination']['has_sources'] else 'No ❌'}, ")
            f.write(f"Hedging={result['base_hallucination']['hedging_count']}\n")
            f.write(f"  Winner:     {'Fine-tuned ✅' if result['finetuned_hallucination']['risk_score'] < result['base_hallucination']['risk_score'] else 'Base ✅'} (lower risk wins)\n\n")
            
            f.write(f"RESPONSE LENGTH:\n")
            f.write(f"  Fine-tuned: {result['finetuned_length']} chars\n")
            f.write(f"  Base:       {result['base_length']} chars\n")
            f.write(f"  Difference: {abs(result['finetuned_length'] - result['base_length'])} chars\n\n")
            
            f.write(f"RESPONSE SIMILARITY: {result['response_similarity']:.4f}\n\n")
        
        # Overall winner with multiple criteria
        f.write("\n" + "="*100 + "\n")
        f.write("🏆 OVERALL ASSESSMENT\n")
        f.write("="*100 + "\n\n")
        
        ft_relevance_wins = sum(1 for r in results if r['finetuned_relevance'] > r['base_relevance'])
        ft_directness_wins = sum(1 for r in results if r['finetuned_q_similarity'] > r['base_q_similarity'])
        ft_specificity_wins = sum(1 for r in results if r['finetuned_specificity']['specificity_score'] > r['base_specificity']['specificity_score'])
        ft_risk_wins = sum(1 for r in results if r['finetuned_hallucination']['risk_score'] < r['base_hallucination']['risk_score'])
        
        f.write(f"Category Winners:\n")
        f.write(f"  Relevance:       {'Fine-tuned' if ft_relevance_wins > len(results)/2 else 'Base'} ({ft_relevance_wins}/{len(results)} wins)\n")
        f.write(f"  Directness:      {'Fine-tuned' if ft_directness_wins > len(results)/2 else 'Base'} ({ft_directness_wins}/{len(results)} wins)\n")
        f.write(f"  Specificity:     {'Fine-tuned' if ft_specificity_wins > len(results)/2 else 'Base'} ({ft_specificity_wins}/{len(results)} wins)\n")
        f.write(f"  Lower Risk:      {'Fine-tuned' if ft_risk_wins > len(results)/2 else 'Base'} ({ft_risk_wins}/{len(results)} wins)\n")
        f.write(f"  Source Citation: {'Fine-tuned' if ft_sources_count > base_sources_count else 'Base'} ({ft_sources_count} vs {base_sources_count})\n\n")
        
        # Calculate overall score
        ft_total_score = ft_relevance_wins + ft_directness_wins + ft_specificity_wins + ft_risk_wins
        base_total_score = (len(results) - ft_relevance_wins) + (len(results) - ft_directness_wins) + \
                          (len(results) - ft_specificity_wins) + (len(results) - ft_risk_wins)
        
        if ft_sources_count > base_sources_count:
            ft_total_score += 1
        else:
            base_total_score += 1
        
        f.write(f"Overall Score:\n")
        f.write(f"  Fine-tuned Model: {ft_total_score} points\n")
        f.write(f"  Base GPT-4o-mini: {base_total_score} points\n\n")
        
        if ft_total_score > base_total_score:
            f.write("🏆 OVERALL WINNER: Fine-tuned Model\n")
            f.write(f"\nKey Strengths:\n")
            if ft_relevance_wins > len(results)/2:
                f.write(f"  ✅ Better topic relevance ({avg_ft_relevance:.4f} vs {avg_base_relevance:.4f})\n")
            if ft_sources_count > base_sources_count:
                f.write(f"  ✅ More source citations ({ft_sources_count} vs {base_sources_count})\n")
            if ft_risk_wins > len(results)/2:
                f.write(f"  ✅ Lower hallucination risk ({avg_ft_risk:.2f} vs {avg_base_risk:.2f})\n")
        else:
            f.write("🏆 OVERALL WINNER: Base GPT-4o-mini\n")
            f.write(f"\nKey Strengths:\n")
            if ft_relevance_wins < len(results)/2:
                f.write(f"  ✅ Better topic relevance ({avg_base_relevance:.4f} vs {avg_ft_relevance:.4f})\n")
            if base_sources_count > ft_sources_count:
                f.write(f"  ✅ More source citations ({base_sources_count} vs {ft_sources_count})\n")
            if ft_risk_wins < len(results)/2:
                f.write(f"  ✅ Lower hallucination risk ({avg_base_risk:.2f} vs {avg_ft_risk:.2f})\n")
    
    print(f"\n✅ Detailed report saved to: {output_file}")

def main():
    print("🚀 Starting model comparison...\n")
    
    # Extract questions and fine-tuned responses
    print("📖 Reading prompts2.txt...")
    qa_pairs = extract_questions_and_responses('prompts2.txt')
    print(f"✅ Found {len(qa_pairs)} question-answer pairs\n")
    
    # Get base model responses and compare
    results = []
    for i, (question, finetuned_response) in enumerate(qa_pairs, 1):
        print(f"\n{'='*100}")
        print(f"Question {i}/{len(qa_pairs)}")
        print(f"{'='*100}")
        print(f"❓ {question}\n")
        
        # Get base model response
        print("🔄 Querying GPT-4o-mini...")
        base_response = get_base_model_response(question)
        
        if base_response:
            print(f"✅ Base model response received ({len(base_response)} chars)")
            
            # Compare responses
            comparison = compare_responses(question, finetuned_response, base_response)
            results.append(comparison)
            
            print(f"\n📊 Quick metrics:")
            print(f"  Relevance: Fine-tuned={comparison['finetuned_relevance']:.4f}, Base={comparison['base_relevance']:.4f}")
            print(f"  Directness: Fine-tuned={comparison['finetuned_q_similarity']:.4f}, Base={comparison['base_q_similarity']:.4f}")
            print(f"  Specificity: Fine-tuned={comparison['finetuned_specificity']['specificity_score']}, Base={comparison['base_specificity']['specificity_score']}")
            print(f"  Hallucination Risk: Fine-tuned={comparison['finetuned_hallucination']['risk_level']}, Base={comparison['base_hallucination']['risk_level']}")
            print(f"  Sources: Fine-tuned={'Yes' if comparison['finetuned_hallucination']['has_sources'] else 'No'}, Base={'Yes' if comparison['base_hallucination']['has_sources'] else 'No'}")
        else:
            print("❌ Failed to get base model response")
    
    # Generate comparison report
    if results:
        print(f"\n\n{'='*100}")
        print("📝 Generating detailed comparison report...")
        print(f"{'='*100}")
        generate_comparison_report(results)
        
        # Print summary to console
        avg_ft_relevance = np.mean([r['finetuned_relevance'] for r in results])
        avg_base_relevance = np.mean([r['base_relevance'] for r in results])
        avg_ft_specificity = np.mean([r['finetuned_specificity']['specificity_score'] for r in results])
        avg_base_specificity = np.mean([r['base_specificity']['specificity_score'] for r in results])
        avg_ft_risk = np.mean([r['finetuned_hallucination']['risk_score'] for r in results])
        avg_base_risk = np.mean([r['base_hallucination']['risk_score'] for r in results])
        ft_sources = sum(1 for r in results if r['finetuned_hallucination']['has_sources'])
        base_sources = sum(1 for r in results if r['base_hallucination']['has_sources'])
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"\n  RELEVANCE:")
        print(f"    Fine-tuned: {avg_ft_relevance:.4f}")
        print(f"    Base:       {avg_base_relevance:.4f}")
        print(f"    Winner:     {'Fine-tuned ✅' if avg_ft_relevance > avg_base_relevance else 'Base ✅'}")
        
        print(f"\n  SPECIFICITY (detail level):")
        print(f"    Fine-tuned: {avg_ft_specificity:.1f} items")
        print(f"    Base:       {avg_base_specificity:.1f} items")
        print(f"    Winner:     {'Fine-tuned ✅' if avg_ft_specificity > avg_base_specificity else 'Base ✅'}")
        
        print(f"\n  HALLUCINATION RISK (lower is better):")
        print(f"    Fine-tuned: {avg_ft_risk:.2f}")
        print(f"    Base:       {avg_base_risk:.2f}")
        print(f"    Winner:     {'Fine-tuned ✅' if avg_ft_risk < avg_base_risk else 'Base ✅'}")
        
        print(f"\n  SOURCE CITATIONS:")
        print(f"    Fine-tuned: {ft_sources}/{len(results)} responses")
        print(f"    Base:       {base_sources}/{len(results)} responses")
        print(f"    Winner:     {'Fine-tuned ✅' if ft_sources > base_sources else 'Base ✅'}")
        
        if avg_ft_relevance > avg_base_relevance:
            print(f"\n🏆 Overall: Fine-tuned Model has better topic relevance")
        else:
            print(f"\n🏆 Overall: Base GPT-4o-mini has better topic relevance")
    else:
        print("\n❌ No results to report")

if __name__ == "__main__":
    main()
