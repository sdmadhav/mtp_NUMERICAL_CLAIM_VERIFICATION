import requests
import json
from typing import List, Dict
import time

class SimpleGoogleRetriever:
    def __init__(self, api_keys: List[str], search_engine_id: str):
        """
        Initialize with multiple API keys for rotation

        Args:
            api_keys: List of Google API keys
            search_engine_id: Custom Search Engine ID
        """
        self.api_keys = api_keys
        self.current_key_index = 0
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def get_current_api_key(self):
        """Get current API key"""
        return self.api_keys[self.current_key_index]

    def rotate_api_key(self):
        """Rotate to next API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        print(f"Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")

    def search_google(self, query: str, num_results: int = 10) -> List[Dict]:
        """
        Search Google and return results with title, snippet, link

        Args:
            query: Search query
            num_results: Number of results (max 10 per request)

        Returns:
            List of search results
        """
        params = {
            'key': self.get_current_api_key(),
            'cx': self.search_engine_id,
            'q': query,
            'num': min(num_results, 10)
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)

            # Check for quota exceeded
            if response.status_code == 429:
                print("Quota exceeded, rotating API key...")
                self.rotate_api_key()
                time.sleep(1)
                return self.search_google(query, num_results)  # Retry with new key

            response.raise_for_status()
            results = response.json()

            documents = []
            if 'items' in results:
                for item in results['items']:
                    documents.append({
                        'title': item.get('title', ''),
                        'snippet': item.get('snippet', ''),
                        'link': item.get('link', '')
                    })

            return documents

        except Exception as e:
            print(f"Error during search: {e}")
            # Try rotating key on error
            self.rotate_api_key()
            return []

    def retrieve_for_claim(self, claim: str, questions: List[str],
                          num_results_per_question: int = 10) -> Dict:
        """
        Retrieve evidence for all questions of a claim

        Args:
            claim: The claim text
            questions: List of questions for this claim
            num_results_per_question: How many results per question

        Returns:
            Dictionary with claim, questions, and evidence
        """
        print(f"\n{'='*80}")
        print(f"Processing claim: {claim[:100]}...")
        print(f"{'='*80}")

        evidence_data = []

        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}/{len(questions)}: {question}")

            # Search Google
            results = self.search_google(question, num_results_per_question)
            print(f"  Retrieved {len(results)} results")

            evidence_data.append({
                'question': question,
                'search_results': results
            })

            # Small delay to avoid rate limiting
            time.sleep(0.5)

        return {
            'claim': claim,
            'evidence': evidence_data
        }


# Your API keys
API_KEYS = [
    "AIzaSyBaRpS1F0r-NcpgH2Nc7DJSKmKjQeSE",
    "AIzaSyDCEL0pSQoY7PYMuvr2ckwhsGgw45i",
    "AIzaSyCeK6qyP6z4kBTkdHJo83XQUnUffvF",
    "AIzaSyCUmAY1grugj7BcLgH09OCv_fMj3sLk",
    "AIzaSyBCcbrBRsBvBnDbkOygI4iQslbkZ_E",
    "AIzaSyBzyo_3rphj8-ezIO6LEzNOo1Fv2A"
]

SEARCH_ENGINE_ID = "351648df707cb49be" #this is sample id use correct one or contact me. 

# Initialize retriever
retriever = SimpleGoogleRetriever(API_KEYS, SEARCH_ENGINE_ID)

with open("claims_with_mmr_and_selected_questions.json", 'r', encoding='utf-8') as f:
    all_claims = json.load(f)

output_file = 'claims_with_evidence.json'

# Load existing progress if file exists
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        all_claims_evidence = json.load(f)
    print(f"Resuming from existing file with {len(all_claims_evidence)} claims already processed")
else:
    all_claims_evidence = []

# Start from where we left off
start_index = len(all_claims_evidence)

# Loop through claim_identifiers
for jind in range(start_index, min(1000, len(all_cat_claims))):
    item = all_cat_claims[jind]
    claim = item['claim']
    questions = [q['question'] for q in item['selected_questions']] #you might have different name of the key but idea is list all the questions/queries generated for which you want to retrieve evidences

    # Retrieve evidence
    claim_evidence = retriever.retrieve_for_claim(
        claim=claim,
        questions=questions, 
        num_results_per_question=10 # max 10 results can be obtained per query. Pagination will be considered as new call to api.
    )

    all_claims_evidence.append(claim_evidence)
    print("index", jind)
    jind += 1
# Save to JSON
output_file = 'claims_with_evidence2.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_claims_evidence, f, ensure_ascii=False, indent=2)

print(f"\n{'='*80}")
print(f"✓ Saved evidence for {len(all_claims_evidence)} claims to {output_file}")
print(f"{'='*80}")
