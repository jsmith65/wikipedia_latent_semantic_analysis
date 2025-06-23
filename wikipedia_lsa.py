import wikipediaapi
import re
import requests
import pprint
import numpy as np
from sklearn.cluster import KMeans

import pdb

# Initialize the Wikipedia API object
wiki = wikipediaapi.Wikipedia( 'Jon Scott Smith (jsmith6503@gmail.com)', 'en' )  # 'en' specifies the English Wikipedia

# Constants
total_pages_to_fetch = 10
pp = pprint.PrettyPrinter( indent = 4  )

# Function to fetch a random Wikipedia article using the MediaWiki API
def fetch_random_article():
    """
    Fetch a random Wikipedia article.
    Returns the title and content of the article.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "random",  # Fetch a random page
        "grnnamespace": 0,     # Only fetch content pages (namespace 0)
        "prop": "extracts",    # Fetch the content
        "explaintext": True,   # Get plain text (no HTML)
        "grnlimit": 1          # Fetch one random page
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    pages = data.get("query", {}).get("pages", {})

    # Extract the first (and only) page's title and content
    for page_id, page_data in pages.items():
        title = page_data.get("title", "Untitled")
        content = page_data.get("extract", "")
        return title, content

# Define a function to fetch the content of a Wikipedia article
def fetch_wikipedia_article(title):
    page = wiki.page(title)
    if not page.exists():
        print(f"Page '{title}' does not exist.")
        return None

    # Retrieve the content and section headings
    content = page.text  # Entire text of the article
    sections = page.sections  # Section objects
    return title, content

def preprocess_text(text):
    """
    Preprocess the text by:
    - Converting to lowercase
    - Removing punctuation
    - Splitting into words
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and non-alphanumeric characters
    text = re.sub(r'[^\w\s]|[\d]', ' ', text)

    # Split into words (tokens)
    words = text.split()
    return words

def count_word_frequencies(words):
    """
    Count the frequencies of each word in a document.
    Returns a dictionary where the key is the word and the value is its frequency.
    """
    word_counts = {}
    for word in words:
        word_counts.setdefault(word, 0)  # Initialize the key with 0 if it doesn’t exist
        word_counts[word] += 1
    return dict(word_counts)

import numpy as np

def build_term_document_matrix(articles_word_counts):
    """
    Build a term-document matrix with TF-IDF weights.

    Parameters:
    - articles_word_counts: A dictionary where keys are article titles and
      values are dictionaries of word counts for each article.

    Returns:
    - term_document_matrix: A NumPy array representing the term-document matrix with TF-IDF weights.
    - terms: A list of unique terms (row indices of the matrix).
    - documents: A list of document titles (column indices of the matrix).
    """
    # Step 1: Collect all unique terms
    terms = set()
    for word_counts in articles_word_counts.values():
        terms.update(word_counts.keys())
    terms = sorted(terms)  # Sort terms for consistent indexing

    # Step 2: Create term and document mappings
    term_to_index = {term: i for i, term in enumerate(terms)}
    document_to_index = {doc: j for j, doc in enumerate(articles_word_counts.keys())}
    documents = list(articles_word_counts.keys())

    # Step 3: Initialize the matrix
    term_document_matrix = np.zeros((len(terms), len(documents)))

    # Step 4: Compute logarithmic TF weights
    log_weights = {}  # Dictionary to store logarithmic weights per document

    for doc, word_counts in articles_word_counts.items():
        log_weights[doc] = {}  # Initialize inner dictionary for this document
        j = document_to_index[doc]  # Get the column index for the document

        for word, count in word_counts.items():
            # Compute logarithmic weight
            log_weight = np.log(1 + count)
            log_weights[doc][word] = log_weight  # Store the weight

    # Step 5: Compute DF and IDF weights
    doc_frequency = {term: 0 for term in terms}  # Initialize DF dictionary

    # Compute document frequency (DF)
    for word_counts in articles_word_counts.values():
        for word in word_counts.keys():
            doc_frequency[word] += 1  # Increment DF for each document containing the word

    # Compute IDF weights
    total_documents = len(documents)
    idf_weights = {
        term: np.log(total_documents / doc_frequency[term])
        for term in terms
    }

    # Step 6: Populate the term-document matrix with TF-IDF weights
    for doc, word_counts in articles_word_counts.items():
        j = document_to_index[doc]  # Get the column index for the document

        for word, count in word_counts.items():
            i = term_to_index[word]  # Get the row index for the term
            tf_weight = log_weights[doc][word]  # Logarithmic TF weight
            idf_weight = idf_weights[word]  # IDF weight

            # Compute TF-IDF and populate the matrix
            term_document_matrix[i, j] = tf_weight * idf_weight

    return term_document_matrix, terms, documents

def select_k_for_energy_retained(singular_values, energy_threshold=0.95):
    """
    Select the smallest k such that the top k singular values retain at least `energy_threshold` energy.

    Parameters:
    - singular_values: A 1D array of singular values from SVD (Sigma).
    - energy_threshold: The fraction of total energy to retain (default: 0.95).

    Returns:
    - k: The number of components to retain.
    """
    # Compute the total energy (sum of squared singular values)
    total_energy = np.sum(singular_values**2)

    # Compute cumulative energy for each k
    cumulative_energy = np.cumsum(singular_values**2)

    # Find the smallest k where cumulative energy >= energy_threshold * total_energy
    k = np.searchsorted(cumulative_energy, energy_threshold * total_energy) + 1
    return k

def kmeans_clustering(data, labels, n_clusters=3):
    """
    Perform K-Means clustering on the given data.

    Parameters:
    - data: A 2D NumPy array where rows are items to cluster.
    - labels: A list of labels corresponding to the rows of the data.
    - n_clusters: The number of clusters to form (default: 3).

    Returns:
    - cluster_assignments: A dictionary mapping each label to its cluster.
    """
    # Initialize and fit the K-Means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)

    # Get cluster assignments
    cluster_assignments = {label: cluster for label, cluster in zip(labels, kmeans.labels_)}

    return cluster_assignments

wiki_pages = dict()

for i in range( total_pages_to_fetch ):

    title, content = fetch_random_article()

    tokens = preprocess_text( content )

    word_frequencies = count_word_frequencies( tokens )

    wiki_pages[ title ] = word_frequencies

#pp.pprint( wiki_pages )

term_document_matrix, terms, documents = build_term_document_matrix( wiki_pages )

print("Term-Document Matrix:")
print(term_document_matrix)

print("\nTerms (Rows):")
print(terms)

print("\nDocuments (Columns):")
print(documents)

U, Sigma, Vt = np.linalg.svd( term_document_matrix, full_matrices=False )

print("U (Left Singular Vectors):")
print(U)

print("\nSigma (Singular Values):")
print(Sigma)

print("\nV^T (Right Singular Vectors):")
print(Vt)

# Select k using the 95% energy retained method
k = select_k_for_energy_retained(Sigma)

# Truncate SVD results to the top k components
U_k = U[:, :k]  # Keep the first k columns of U
Sigma_k = np.diag(Sigma[:k])  # Convert top k singular values to a diagonal matrix
Vt_k = Vt[:k, :]  # Keep the first k rows of Vt

print(f"Selected k: {k}")
print("Truncated U_k:")
print(U_k)
print("\nTruncated Sigma_k:")
print(Sigma_k)
print("\nTruncated Vt_k:")
print(Vt_k)

# Cluster the Terms Matrix
term_cluster_assignments = kmeans_clustering(U_k, terms, k)

print("Cluster Assignments (Terms):")
for term, cluster in term_cluster_assignments.items():
    print(f"{term}: Cluster {cluster}")

# Cluster the Document Matrix
V_k = Vt_k.T  # Transpose to get documents as rows

# Perform K-Means clustering on documents
document_cluster_assignments = kmeans_clustering(V_k, documents, k)

print("Cluster Assignments (Documents):")
for document, cluster in document_cluster_assignments.items():
    print(f"{document}: Cluster {cluster}")