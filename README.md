# InfoBOT : News_research_tool_using_LLM

 InfoBOT is a user-friendly news research tool designed for effortless information retrieval. Users can input article URLs and ask questions to receive relevant insights from the stock market and financial domain.

 ![](download.png)
 
## Features

- Load upto three URLs or upload text files containing URLs to fetch article content.
- Process article content through LangChain's SeleniumLoader Loader
- Constructs an embedding vector using OpenAI's embeddings and leverage FAISS, a powerful similarity search library, to enable swift and effective retrieval of relevant information
- Interact with the LLM's (Chatgpt) by inputting queries and receiving answers along with source URLs.

## Project Structure

- main.py: The main Streamlit application script.
- requirements.txt: A list of required Python packages for the project.
- faiss_store_openai.pkl: A pickle file to store the FAISS index.
- .env: Configuration file for storing your OpenAI API key.


## Installation

1.Clone this repository to your local machine using:

```bash
  
```
2.Navigate to the project directory:

```bash
  cd BreadcrumbsNews_research_app_using_LLM
```
3. Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```
4.Set up your OpenAI API key by creating a .env file in the project root and adding your API

```bash
  OPENAI_API_KEY=your_api_key_here
