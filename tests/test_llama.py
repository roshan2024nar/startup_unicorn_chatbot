"""
tests/test_llama.py

Manual tests for LlamaIndex structured query engine.

Run:
    python -m tests.test_llama
"""

from stores.llama_store import query_llama, is_error_response


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run_test(query: str, check_fn=None, label: str = "") -> bool:
    """Run a structured query and optionally validate the result."""
    ans = query_llama(query)

    if is_error_response(ans):
        print(f"FAIL  {label or query}")
        print(f"  error: {ans[:150]}\n")
        return False

    passed = check_fn(ans) if check_fn else True
    status = "PASS" if passed else "CHECK"

    print(f"{status}  {label or query}")
    print(f"  answer: {ans[:200]}\n")

    return passed


# ---------------------------------------------------------------------------
# Test Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("LlamaIndex Structured Query Tests\n")

    results = []

    # Investor queries
    results.append(run_test(
        "Who invested in Razorpay?",
        check_fn=lambda a: any(w in a.lower()
                               for w in ["tcv", "gic", "sequoia", "tiger"]),
        label="Razorpay investors",
    ))

    results.append(run_test(
        "Who invested in Swiggy?",
        check_fn=lambda a: any(w in a.lower()
                               for w in ["dst", "softbank", "tencent", "prosus"]),
        label="Swiggy investors",
    ))

    results.append(run_test(
        "Who invested in PhonePe?",
        label="PhonePe investors",
    ))

    # Funding / valuation queries
    results.append(run_test(
        "How much total funding has Razorpay raised?",
        check_fn=lambda a: "$" in a or "million" in a.lower() or "usd" in a.lower(),
        label="Razorpay total funding",
    ))

    results.append(run_test(
        "What is the valuation of Swiggy?",
        check_fn=lambda a: "billion" in a.lower() or "$" in a,
        label="Swiggy valuation",
    ))

    results.append(run_test(
        "Which unicorn has the highest valuation?",
        check_fn=lambda a: len(a) > 10,
        label="Highest valuation unicorn",
    ))

    # Aggregation queries
    results.append(run_test(
        "How many fintech startups are in Bangalore?",
        check_fn=lambda a: any(c.isdigit() for c in a),
        label="Count fintech in Bangalore",
    ))

    results.append(run_test(
        "How many unicorns joined in 2021?",
        check_fn=lambda a: any(c.isdigit() for c in a),
        label="Unicorns joined in 2021",
    ))

    results.append(run_test(
        "Which sector has the most unicorns?",
        label="Sector with most unicorns",
    ))

    results.append(run_test(
        "Which city has the most unicorns?",
        label="City with most unicorns",
    ))

    # Investor-based filtering
    results.append(run_test(
        "List all startups backed by SoftBank",
        check_fn=lambda a: len(a) > 20,
        label="Startups backed by SoftBank",
    ))

    results.append(run_test(
        "Which startups were founded before 2010?",
        check_fn=lambda a: len(a) > 10,
        label="Startups founded before 2010",
    ))

    passed = sum(results)
    total  = len(results)

    print(f"Results: {passed}/{total} {'PASS' if passed == total else 'CHECK'}")