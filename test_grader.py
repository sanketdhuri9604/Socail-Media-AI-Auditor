"""Quick test to verify grader math passes validator checks."""
from models import AuditAction
from server.grader import grade_detailed
from server.tasks import TASKS

def test_perfect_scores():
    print("=== PERFECT SCORES (all correct, matching explanations) ===")
    all_ok = True
    for task_id, task in TASKS.items():
        gt = task["ground_truth"]
        action = AuditAction(
            hallucination_detected=gt["hallucination"],
            hallucination_explanation=gt["hallucination_reason"],
            bias_detected=gt["bias"],
            bias_explanation=gt["bias_reason"],
            alignment_violated=gt["alignment_violated"],
            alignment_explanation=gt["alignment_reason"],
            memory_consistent=gt["memory_consistent"],
            memory_explanation=gt["memory_reason"],
            overall_verdict=gt["verdict"],
            confidence=0.7,
        )
        result = grade_detailed(action, gt)
        reward = result["reward"]
        in_range = 0.0 < reward < 1.0
        print(f"  {task_id}: reward={reward}  in_range={in_range}")
        if not in_range:
            all_ok = False

        # Check breakdown values
        for k, v in result["breakdown"].items():
            if isinstance(v, (int, float)):
                bd_ok = 0.0 < float(v) < 1.0
                if not bd_ok:
                    print(f"    FAIL: breakdown[{k}]={v} NOT in (0,1)")
                    all_ok = False

    print(f"  ALL IN RANGE: {all_ok}\n")
    return all_ok


def test_non_constant():
    print("=== NON-CONSTANT CHECK (same action, different tasks) ===")
    fixed = AuditAction(
        hallucination_detected=True,
        hallucination_explanation="x",
        bias_detected=True,
        bias_explanation="x",
        alignment_violated=True,
        alignment_explanation="x",
        memory_consistent=False,
        memory_explanation="x",
        overall_verdict="remove",
        confidence=0.95,
    )
    scores = {}
    for task_id, task in TASKS.items():
        r = grade(fixed, task["ground_truth"])
        scores[task_id] = r["reward"]
        print(f"  {task_id}: {r['reward']}")

    unique = len(set(scores.values()))
    print(f"  Unique scores: {unique} (must be > 1)")
    return unique > 1


def test_partial_credit():
    print("=== PARTIAL CREDIT CHECK (varying correctness) ===")
    gt = TASKS["easy"]["ground_truth"]
    scores = []

    # 0/5 correct
    a0 = AuditAction(
        hallucination_detected=False, hallucination_explanation="wrong",
        bias_detected=True, bias_explanation="wrong",
        alignment_violated=False, alignment_explanation="wrong",
        memory_consistent=False, memory_explanation="wrong",
        overall_verdict="safe", confidence=0.5,
    )
    r0 = grade(a0, gt)["reward"]
    scores.append(r0)
    print(f"  0/5 correct: {r0}")

    # 2/5 correct
    a2 = AuditAction(
        hallucination_detected=True, hallucination_explanation="fabricated claim",
        bias_detected=False, bias_explanation="no bias",
        alignment_violated=True, alignment_explanation="violates rules",
        memory_consistent=False, memory_explanation="wrong",
        overall_verdict="safe", confidence=0.5,
    )
    r2 = grade(a2, gt)["reward"]
    scores.append(r2)
    print(f"  2/5 correct: {r2}")

    # 4/5 correct
    a4 = AuditAction(
        hallucination_detected=True, hallucination_explanation="fabricated medical claim",
        bias_detected=False, bias_explanation="no bias present",
        alignment_violated=True, alignment_explanation="violates medical claim rules",
        memory_consistent=True, memory_explanation="recurring cure narratives",
        overall_verdict="safe", confidence=0.7,
    )
    r4 = grade(a4, gt)["reward"]
    scores.append(r4)
    print(f"  4/5 correct: {r4}")

    # 5/5 correct with good explanations
    a5 = AuditAction(
        hallucination_detected=True,
        hallucination_explanation=gt["hallucination_reason"],
        bias_detected=False,
        bias_explanation=gt["bias_reason"],
        alignment_violated=True,
        alignment_explanation=gt["alignment_reason"],
        memory_consistent=True,
        memory_explanation=gt["memory_reason"],
        overall_verdict="remove",
        confidence=0.7,
    )
    r5 = grade(a5, gt)["reward"]
    scores.append(r5)
    print(f"  5/5 correct: {r5}")

    monotonic = all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))
    print(f"  Monotonically increasing: {monotonic}")
    return monotonic


if __name__ == "__main__":
    ok1 = test_perfect_scores()
    ok2 = test_non_constant()
    ok3 = test_partial_credit()

    print("\n" + "=" * 50)
    if ok1 and ok2 and ok3:
        print("ALL CHECKS PASSED ✅")
    else:
        print("SOME CHECKS FAILED ❌")
        if not ok1:
            print("  - Perfect scores range check failed")
        if not ok2:
            print("  - Non-constant check failed")
        if not ok3:
            print("  - Partial credit check failed")
