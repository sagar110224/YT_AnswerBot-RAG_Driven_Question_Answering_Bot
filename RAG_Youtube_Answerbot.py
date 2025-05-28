from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from urllib.parse import urlparse, parse_qs

class RAG_Youtube:
    def __init__(self):
        load_dotenv()
        self.parser=StrOutputParser()
        self.llm=ChatOpenAI()
        self.prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.
        {context} 
        Question: {question}
        """,
        input_variables = ['context', 'question']
        )
    
    def check_en_transcript(self,video_id):
        #getting list of available transcript languages
        lst=YouTubeTranscriptApi().list(video_id) 
        transcript_lst=[]
        for l in lst:
            l=str(l)
            transcript_lst.append(l[0:2])
        
        if 'en' in transcript_lst:
            return lst,True
        else:
            return lst,False
    
    def translate_to_en_transcript(self,transcript_list): #To generate english transcript if it is not available
        translated=""
        for l in transcript_list:
            translated=l.translate("en").fetch()
        return translated
    
    def get_en_transcript(self,video_id):#Extracting english transcript if it is available
        return YouTubeTranscriptApi.get_transcript(video_id,languages=['en'])
    
    def create_en_transcript_corpus(self,transcript_dict,en_present): #Creating a corpus of words spoken in the video
        corpus=""
        if en_present==True:
            for x in transcript_dict:
                corpus=corpus+x['text']+" "
        elif en_present==False:
            for x in transcript_dict:
                corpus=corpus+x.text+" " 
        return corpus

    
    def create_transcript_corpus(self,video_id): #Function to get transcript from the video_id provided
        transcript_list,en_present=self.check_en_transcript(video_id)
        if en_present==False:
            transcript=self.translate_to_en_transcript(transcript_list)
        else:
            transcript=self.get_en_transcript(video_id)
        transcript_corpus= self.create_en_transcript_corpus(transcript,en_present)
        return transcript_corpus
    
    def text_splitting(self,corpus,chunk_size=1000,chunk_overlap=100): #splitting corpus into chunks
        splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        return splitter.create_documents([corpus])
    
    def create_vector_store(self,chunks): #Creating a list of documents with each documents storing a chunk of transcript
        embeddings=OpenAIEmbeddings()
        return FAISS.from_documents(chunks,embeddings)
    
    def initialize_retriever(self,vector_store,k=4): 
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k}) 
        return retriever
    
    def format_docs(self,retrieved_docs):
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context_text

    def extract_video_id(self,url): #to extract videoid from URL
        query=urlparse(url).query
        params=parse_qs(query)
        return params.get("v",[None])[0]
    
    def main(self,url): #Creating the chain using lanchain
        try:
            video_id=self.extract_video_id(url)
            if video_id==None:
                print("Videoid not present")
                return None
            else:    
                corpus=self.create_transcript_corpus(video_id)
                chunks=self.text_splitting(corpus)
                vector_store=self.create_vector_store(chunks)
                retriever=self.initialize_retriever(vector_store)
                parallal_chain=RunnableParallel(
                {'context':retriever|RunnableLambda(self.format_docs),
                'question':RunnablePassthrough()
                })
                main_chain=parallal_chain|self.prompt|self.llm|self.parser
                return main_chain
        except:
            print("Incorrect URL")
            return None
    
    def run(self,video_id,query): #calling the chain
        chain=self.main(video_id)
        if chain!=None:
            return chain.invoke(query)
        else:
            return None

        
if __name__=='__main__':
    RAG=RAG_Youtube()
    url='https://www.youtube.com/watch?v=J5_-l7WIO_w&list=PLKnIA16_RmvaTbihpo4MtzVm4XOQa0ER0&index=17&ab_channel=CampusX'
    query="What is indexing"
    output=RAG.run(url,query)
    print(output)

