import uuid
import time
import textwrap
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph import chatbot


TEST_CASES = [
    {
        "category": "1. REAL-TIME SEARCH",
        "description": "Should call get_stock_price + web_news_search and use real, sourced facts — not memorized numbers.",
        "query": "what is latets news of dilhi protest",
        "expected_tool": True,
    }
]


def _print_wrapped(label: str, text: str, width: int = 88, indent: str = "  "):
    print(f"{label}")
    for line in textwrap.wrap(text, width=width) or [""]:
        print(f"{indent}{line}")


def run_single_turn(thread_id: str, query: str):
    """Runs one message through the graph. Returns (event_log, elapsed_seconds)
    where event_log is a list of dicts describing exactly what each node did,
    for later inspection/printing."""
    config = {"configurable": {"thread_id": thread_id}}
    initial_input = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "iteration": 0,
        "max_iteration": 2,
    }

    event_log = []
    pending_retry_reason = None  # set right after a needs_more_data verdict
    start_time = time.time()

    for event in chatbot.stream(initial_input, config=config, stream_mode="updates"):
        for node_name, node_update in event.items():
            entry = {"node": node_name}

            if node_name == "chat_node":
                if pending_retry_reason:
                    entry["label"] = "chat_node [RE-SEARCH RETRY]"
                    entry["retry_reason"] = pending_retry_reason
                    pending_retry_reason = None
                else:
                    entry["label"] = "chat_node"

                messages = node_update.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    tool_calls = getattr(last_msg, "tool_calls", None)
                    if tool_calls:
                        entry["tool_calls"] = [
                            f"{tc['name']}({tc.get('args', {})})" for tc in tool_calls
                        ]
                    elif isinstance(last_msg, AIMessage) and last_msg.content:
                        entry["answer_preview"] = str(last_msg.content)[:150]

            elif node_name == "tools":
                entry["label"] = "tools"
                messages = node_update.get("messages", [])
                if messages:
                    entry["tool_output_preview"] = str(messages[-1].content)[:300]

            elif node_name == "evaluate_answer":
                entry["label"] = "evaluate_answer"
                entry["verdict"] = node_update.get("evaluation")
                entry["improvement_type"] = node_update.get("improvement_type")
                entry["feedback"] = node_update.get("feedback")
                entry["iteration_after"] = node_update.get("iteration")
                if node_update.get("evaluation") == "needs_improvement" and node_update.get("improvement_type") == "needs_more_data":
                    pending_retry_reason = node_update.get("feedback")

            elif node_name == "optimize_answer":
                entry["label"] = "optimize_answer"
                entry["answer_preview"] = str(node_update.get("answer", ""))[:150]

            event_log.append(entry)

            # live progress line
            label = entry.get("label", node_name)
            extra = ""
            if "tool_calls" in entry:
                extra = f" — calls: {entry['tool_calls']}"
            elif "verdict" in entry:
                extra = f" — {entry['verdict']}" + (f" ({entry['improvement_type']})" if entry.get("improvement_type") else "")
            print(f"  ⚡ [{label}]{extra}")

    elapsed = round(time.time() - start_time, 2)
    return event_log, elapsed


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

        all_events = []
        total_elapsed = 0.0

        try:
            for turn_idx, query in enumerate(turns, start=1):
                print(f"\n💬 Turn {turn_idx}: \"{query}\"\n")
                event_log, elapsed = run_single_turn(thread_id, query)
                all_events.extend(event_log)
                total_elapsed += elapsed
        except Exception as e:
            print(f"❌ ERROR on test {idx}: {e}")
            results.append({"test": test["category"], "error": str(e)})
            continue

        final_state = chatbot.get_state({"configurable": {"thread_id": thread_id}}).values
        final_answer = final_state.get("answer", "(no answer captured)")

        tools_called = sorted({
            call.split("(")[0]
            for e in all_events if "tool_calls" in e
            for call in e["tool_calls"]
        })
        node_path = [e.get("label", e["node"]) for e in all_events]
        optimized = any(e["node"] == "optimize_answer" for e in all_events)
        retried_search = any("RE-SEARCH RETRY" in e.get("label", "") for e in all_events)

        print("\n" + "-" * 90)
        print("📊 SUMMARY")
        print(f"  • Node path         : {' → '.join(node_path)}")
        print(f"  • Tool(s) called    : {tools_called if tools_called else 'none'}"
              + (f"  (expected a tool: {'yes' if test['expected_tool'] else 'no'})" if test["expected_tool"] is not None else ""))
        print(f"  • Optimizer ran     : {'✅ yes' if optimized else '❌ no (approved on first draft)'}")
        print(f"  • Re-searched?      : {'✅ yes — evaluator sent it back for more data' if retried_search else 'ℹ️ no'}")
        print(f"  • Final verdict     : {final_state.get('evaluation')}")
        print(f"  • Final iteration   : {final_state.get('iteration')} / {final_state.get('max_iteration')}")
        print(f"  • Total time        : {total_elapsed}s")

        # show raw tool output(s) so search/stock data quality can be eyeballed
        tool_previews = [e["tool_output_preview"] for e in all_events if "tool_output_preview" in e]
        for i, preview in enumerate(tool_previews, start=1):
            _print_wrapped(f"  • Raw tool output {i} :", preview)

        _print_wrapped("  • FINAL ANSWER      :", final_answer)
        print("-" * 90)

        results.append({
            "test": test["category"],
            "tools_called": tools_called,
            "optimized": optimized,
            "retried_search": retried_search,
            "final_answer": final_answer,
            "elapsed": total_elapsed,
        })

    print("\n" + "=" * 90)
    print("🎉 ALL TESTS COMPLETE")
    print("=" * 90 + "\n")

    return results


if __name__ == "__main__":
    run_test_suite()