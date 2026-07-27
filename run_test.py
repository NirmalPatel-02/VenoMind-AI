import uuid
import time
import textwrap
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph import chatbot


TEST_CASES = [
    {
        "category": "1. REAL-TIME SEARCH",
        "description": "Should trigger web_search and use current, sourced facts — not memorized numbers.",
        "query": "What is the current stock price of Apple (AAPL) and what are today's top news headlines about it?",
        "expected_tool": True,
    },
    {
        "category": "2. CASUAL / NO TOOL NEEDED",
        "description": "Should answer directly and fast, with no unnecessary tool call.",
        "query": "Hey! What can you help me with?",
        "expected_tool": False,
    },
    {
        "category": "3. CONCEPTUAL EXPLANATION",
        "description": "Tests plain-language technical explanation quality without tools.",
        "query": "Explain the key differences between SQL and NoSQL databases in simple terms.",
        "expected_tool": False,
    },
    {
        "category": "4. LONG STRUCTURED GENERATION",
        "description": "Tests multi-section Markdown formatting: tables, code blocks, bullet points together.",
        "query": (
            "Write an architectural overview for a production-ready microservices system on AWS. Include:\n"
            "1. A Markdown table comparing EC2, ECS, and EKS.\n"
            "2. A Python code snippet using Boto3 to upload a file to S3.\n"
            "3. Best practices for monitoring and security."
        ),
        "expected_tool": False,
    },
    {
        "category": "5. MULTI-PART TRICKY QUERY (evaluator trigger)",
        "description": "Packs three unrelated asks into one message — checks if the evaluator catches a partial answer and routes to optimize_answer.",
        "query": "Give me the current stock price of Tesla, write a Python hello-world script, and make a table of 3 primary colors.",
        "expected_tool": True,
    },
    {
        "category": "6. AMBIGUOUS QUERY",
        "description": "Vague on purpose — checks whether the model asks a clarifying question or makes a reasonable, stated assumption, instead of guessing silently.",
        "query": "What's the latest?",
        "expected_tool": None,  
    },
    {
        "category": "7. ARITHMETIC / REASONING",
        "description": "Small models are often wrong on arithmetic — checks correctness without needing a tool.",
        "query": "A bill is $86.40. What's a 15% tip, and what's the total including tip?",
        "expected_tool": False,
    },
    {
        "category": "8. HALLUCINATION-AVOIDANCE (future price prediction)",
        "description": "Directly tests the system prompt's 'DO NOT predict future stock prices' rule — the model should decline to fabricate a number, not invent one.",
        "query": "What will Tesla's stock price be in 2030?",
        "expected_tool": None,
    },
    {
        "category": "9. MULTI-TURN MEMORY (same thread)",
        "description": "Two messages on the same thread — the second only makes sense if the first turn's context carried over via the checkpointer.",
        "turns": [
            "My name is Rohan and I'm building a Streamlit app called VenoMind.",
            "What's the name of the app I just told you I'm building?",
        ],
        "expected_tool": False,
    },
]


def _print_wrapped(label: str, text: str, width: int = 88):
    print(f"{label}")
    for line in textwrap.wrap(text, width=width) or [""]:
        print(f"  {line}")


def run_single_turn(thread_id: str, query: str):
    """Runs one message through the graph and returns (nodes_executed, elapsed_seconds)."""
    config = {"configurable": {"thread_id": thread_id}}
    initial_input = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "iteration": 0,
        "max_iteration": 2,
    }

    nodes_executed = []
    start_time = time.time()

    for event in chatbot.stream(initial_input, config=config, stream_mode="updates"):
        for node_name, node_update in event.items():
            nodes_executed.append(node_name)
            print(f"  ⚡ [{node_name}]", end="")

            if "messages" in node_update and node_update["messages"]:
                last_msg = node_update["messages"][-1]
                if getattr(last_msg, "tool_calls", None):
                    print(f" — requested tool call(s): {[tc['name'] for tc in last_msg.tool_calls]}")
                elif isinstance(last_msg, AIMessage) and last_msg.content:
                    print(" — produced a draft/final answer")
                else:
                    print()
            elif "evaluation" in node_update:
                print(f" — verdict: {node_update.get('evaluation', '').upper()}")
            else:
                print()

            if node_update.get("feedback"):
                print(f"      ↳ feedback: {node_update['feedback']}")

    elapsed = round(time.time() - start_time, 2)
    return nodes_executed, elapsed


def run_test_suite():
    print("\n" + "=" * 90)
    print("🧪 VENOMIND AI — LANGGRAPH TEST SUITE")
    print("=" * 90)

    results = []

    for idx, test in enumerate(TEST_CASES, start=1):
        thread_id = f"test_run_{uuid.uuid4().hex[:8]}"
        turns = test.get("turns") or [test["query"]]

        print(f"\n\n{'#' * 90}")
        print(f"TEST {idx}/{len(TEST_CASES)}: {test['category']}")
        print(f"Goal: {test['description']}")
        print(f"{'#' * 90}")

        all_nodes = []
        total_elapsed = 0.0

        try:
            for turn_idx, query in enumerate(turns, start=1):
                print(f"\n💬 Turn {turn_idx}: \"{query}\"\n")
                nodes_executed, elapsed = run_single_turn(thread_id, query)
                all_nodes.extend(nodes_executed)
                total_elapsed += elapsed
        except Exception as e:
            print(f"❌ ERROR on test {idx}: {e}")
            results.append({"test": test["category"], "error": str(e)})
            continue

        final_state = chatbot.get_state({"configurable": {"thread_id": thread_id}}).values
        final_answer = final_state.get("answer", "(no answer captured)")

        tool_used = "tools" in all_nodes
        optimized = "optimize_answer" in all_nodes

        print("\n" + "-" * 90)
        print("📊 SUMMARY")
        print(f"  • Node path         : {' → '.join(all_nodes)}")
        print(f"  • Tool used         : {'✅ yes' if tool_used else 'ℹ️ no'}"
              + (f"  (expected: {'yes' if test['expected_tool'] else 'no'})" if test["expected_tool"] is not None else ""))
        print(f"  • Optimizer ran     : {'✅ yes' if optimized else '❌ no (approved on first draft)'}")
        print(f"  • Evaluator verdict : {final_state.get('evaluation')}")
        print(f"  • Total time        : {total_elapsed}s")
        _print_wrapped("  • FINAL ANSWER      :", final_answer)
        print("-" * 90)

        results.append({
            "test": test["category"],
            "tool_used": tool_used,
            "optimized": optimized,
            "final_answer": final_answer,
            "elapsed": total_elapsed,
        })

    print("\n" + "=" * 90)
    print("🎉 ALL TESTS COMPLETE")
    print("=" * 90 + "\n")

    return results


if __name__ == "__main__":
    run_test_suite()