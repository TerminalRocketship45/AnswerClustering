def find_failure_modes(solution, hint=None):

    # (find-criteria solution)
    criteria = find_criteria(solution)

    # (find-if #'(lambda (p) (isa-p p '(goal barrier cause))) (gbsmcontext solution))
    purpose = next(
        (p for p in gbsm_context(solution) if p.ptype in ("goal", "barrier", "cause")),
        None
    )

    # (with-output-to-string (stream) ...)
    lines = []

    # (pr "~a" (explain-purpose purpose))
    lines.append(explain_purpose(purpose))

    # (pr "~2%I want you to identify the different ways that we can fail to ")
    lines.append("\n\nI want you to identify the different ways that we can fail to ")

    # (case (ptype purpose)
    #   (goal    (pr "achieve the goal ~a" (name purpose)))
    #   ((barrier cause) (pr "Overcome the problem ~a" (name purpose))))
    if purpose.ptype == "goal":
        lines.append(f"achieve the goal {purpose.name}")
    elif purpose.ptype in ("barrier", "cause"):
        lines.append(f"Overcome the problem {purpose.name}")

    # (pr "~2%These are the criteria that a good solution for this ~a would satisfy:" (tre (ptype purpose)))
    lines.append(f"\n\nThese are the criteria that a good solution for this {purpose.ptype} would satisfy:")

    # (loop for c in criteria do (pr "~%- ~a ~a ~a" (eid c) (name c) (or (description c) "")))
    for c in criteria:
        lines.append(f"\n- {c.eid} {c.name} {c.description or ''}")

    # (pr "~2%I want you to enumerate, for each of the criteria, the ways that we can fail to achieve these criteria.")
    lines.append("\n\nI want you to enumerate, for each of the criteria, the ways that we can fail to achieve these criteria.")

    # (pr "~2%And I want you to tell me, for each failure mode, whether that failure is highly likely to occur if we use the following solution:")
    lines.append("\n\nAnd I want you to tell me, for each failure mode, whether that failure is highly likely to occur if we use the following solution:")

    # (pr "~%~a ~a" (name solution) (or (description solution) ""))
    lines.append(f"\n{solution.name} {solution.description or ''}")

    # (unless (empty hint) (pr "~2%Do your best to incorporate this hint when defining the failure modes: ~a" hint))
    if hint:
        lines.append(f"\n\nDo your best to incorporate this hint when defining the failure modes: {hint}")

    # (pr "~2%Give me your response as a JSON structure ...")
    lines.append("\n\nGive me your response as a JSON structure that gives the list of failure modes you found, as follows:")
    lines.append('\n```json')
    lines.append('\n[')
    lines.append('\n{\n  "type": "failure",')
    lines.append('\n  "criterionName": <the name for the criterion affected by the failure mode>')
    lines.append('\n  "criterionID": <the ID for the criterion affected by the failure mode>')
    lines.append('\n  "description": <a 2 or 3 sentence description of this proposed failure mode>,')
    lines.append('\n  "name": <a 4 to 6 word name for the failure mode>')
    lines.append('\n  "risk": <say high if the solution is at high risk for this failure, and low if the solution is at low risk>')
    lines.append('\n  "rationale": <a 2 or 3 sentence description of WHY you gave that risk rating for that failure mode>')
    lines.append('\n}]```')

    prompt = "".join(lines)

    # (askLLM prompt)
    response = askLLM(prompt)

    # (parse-llm-output response)
    sexp = parse_llm_output(response)

    return sexp
