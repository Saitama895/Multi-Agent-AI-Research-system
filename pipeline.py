from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain
from langchain.messages import HumanMessage

def run_research_pipeline(topic : str) -> dict:
    state = {}

    # search agent working
    myPrint("step 1 - search agent is working ...")

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [
            HumanMessage(
                content=f"Find recent, reliable and detailed information about: {topic}"
            )
        ]
    })
    state["search_results"] = search_result['messages'][-1].content

    print("\n search result ", state['search_results'])

    # Reader agent working
    myPrint("step 2 - Reader agent is scraping top resources ...")

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages" : [HumanMessage(content=
            f"""Based on the following search results about '{topic}', 
            pick the most relevant URL and scrape it for deeper content.\n\n"
            Search Results:\n{state['search_results'][:800]}"""
        )]
    })

    state['scraped_content'] = reader_result['messages'][-1].content

    print("\nScraped Content\n", state["scraped_content"])

    # Writer chain
    myPrint("step 3 - Writer is drafting the report ...")

    research_combined = (
        f"Search Result : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT: \n {state['scraped_content']}"
    )
    state["report"] = writer_chain.invoke({
        "topic" : topic,
        "research": research_combined
    })

    print("\n Final Report\n", state["report"])

    # critic report
    myPrint("step 4 - Critic is reviewing the report")

    state["feedback"] = critic_chain.invoke({
        "report" : state["report"]
    })

    print("\n critic report \n", state['feedback'])

    return state

def myPrint(text: str):
    print("\n" + " ="*50)
    print(text)
    print("="*50)

if __name__ == "__main__":
    topic = input("\n Enter a research topic: ")
    run_research_pipeline(topic)