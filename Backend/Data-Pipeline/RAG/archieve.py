import requests 
import xml.etree.ElementTree as ET
from nltk.stem import WordNetLemmatizer
import nltk

# user_input = "AI Cancer Detection"

# response = requests.get('https://export.arxiv.org/api/query?search_query=all:AI+AND+all:cancer+AND+all:detection') 
# print(response.text)

query = input("Enter your Query: ")

nltk.download('wordnet')

def Normalization_of_query (query):
    query = query.split()
    lemmatizer = WordNetLemmatizer()
    filtered_words = [lemmatizer.lemmatize(query) for word in filtered_words]
    return filtered_words
print(Normalization_of_query(query))
# def fetch_arxiv_xml(normalized_query):
#     words = query.split()
    
#     formatted_words = ["all:" + word for word in words]
#     search_query = " AND ".join(formatted_words)
    
#     search_query = search_query.replace(" ","+")

#     url = f"http://export.arxiv.org/api/query?search_query={search_query}&max_results=10"

#     response = requests.get(url)

#     return response.text




# print(fetch_arxiv_xml(query))

# xml_data = fetch_arxiv_xml(query)

# root = ET.fromstring(xml_data)

# print(root)

# entries = root.findall("{http://www.w3.org/2005/Atom}entry")
# print(len(entries))

# print(entries[0])
# First 

# Normalization on data 
# Convert Query to lower Case
# split Query into words

# second

# 