from evals.run import evaluate


def test_authored_retrieval_regressions():
    report = evaluate()
    assert report["passed"] == report["cases"]
