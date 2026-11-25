import html
import re
from urllib.parse import quote_plus
import numpy as np
import pandas as pd
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import norm

COMPETENCY_TAXONOMY = {
    "Data & BI Tools": {
        "python": ["python", "pandas", "numpy", "streamlit"],
        "sql": ["sql", "postgresql", "mysql", "sqlite", "db"],
        "r": ["r"],
        "excel": ["excel", "spreadsheets", "vlookup", "pivot tables", "macros"],
        "power bi": ["power bi", "dax", "power query"],
        "tableau": ["tableau"],
        "dashboards & reporting": ["dashboard", "dashboards", "reporting", "reports", "bi", "business intelligence"]
    },
    "Analytics & Forecasting": {
        "data analysis": ["data analysis", "data analyst", "data analytics", "data & reporting", "analyze", "analytical"],
        "data cleaning & etl": ["data cleaning", "etl", "data preparation", "wrangling", "data quality"],
        "forecasting & modeling": ["forecasting", "forecasts", "modeling", "predictive", "demand planning"],
        "statistics & metrics": ["statistics", "statistical", "kpi", "kpis", "metrics", "variance"]
    },
    "HR & People Analytics": {
        "people & hr analytics": ["hr analytics", "people analytics", "human resources", "hr", "shrm", "hrci"],
        "workforce & headcount": ["workforce planning", "workforce", "headcount", "staffing"],
        "payroll & timekeeping": ["payroll", "timekeeping", "compensation", "work-rule compliance"],
        "recruiting & onboarding": ["recruiting", "talent acquisition", "onboarding", "training"],
        "retention & engagement": ["retention", "turnover", "attrition", "employee engagement", "performance management"]
    },
    "Workforce & Labor Ops": {
        "scheduling & shifts": ["scheduling", "schedules", "shift", "crew coordination"],
        "labor & cost control": ["labor", "labor cost", "cost reduction", "operating expenses", "budgeting", "utilization"],
        "operations & logistics": ["operations", "logistics", "event staffing", "vendor management", "service operations"]
    },
    "Process & Compliance": {
        "process improvement": ["process improvement", "efficiency", "lean", "six sigma", "workflow"],
        "compliance & quality": ["compliance", "work rules", "quality assurance", "audit", "recordkeeping", "sop"]
    },
    "Leadership & Strategy": {
        "leadership & supervision": ["leadership", "supervised", "supervisor", "manager", "mentoring", "coaching", "lead"],
        "communication & decks": ["communication", "present", "presentations", "clear reports", "stakeholder", "client relations"],
        "strategy & consulting": ["strategic planning", "strategic", "consulting", "advisory", "change management", "cross-functional"]
    }
}

def extract_text_from_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    file_name = uploaded_file.name.lower()
    text = ""

    if file_name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    elif file_name.endswith(".docx"):
        doc = Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file_name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    return clean_text(text)

def clean_text(text: str) -> str:
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.splitlines()]
    cleaned = "\n".join([line for line in lines if line])
    return cleaned.strip()

def extract_skills_by_category(text: str) -> dict:
    text_lower = text.lower()
    found = {}
    for category, skill_map in COMPETENCY_TAXONOMY.items():
        matched_canonical = set()
        for canonical_skill, aliases in skill_map.items():
            for alias in aliases:
                if re.search(rf'\b{re.escape(alias)}\b', text_lower):
                    matched_canonical.add(canonical_skill)
                    break
        found[category] = matched_canonical
    return found

def build_highlighted_html(
    text: str,
    green_canonical: list[str],
    secondary_canonical: list[str],
    secondary_color: str = "blue"
) -> str:
    escaped_text = html.escape(text)
    alias_to_style = {}

    for category, skill_map in COMPETENCY_TAXONOMY.items():
        for canonical_skill, aliases in skill_map.items():
            if canonical_skill in green_canonical:
                for alias in aliases:
                    alias_to_style[alias] = (
                        "background-color: rgba(0, 204, 150, 0.28); "
                        "border-bottom: 2px solid #00CC96; "
                        "padding: 1px 4px; border-radius: 4px; font-weight: 600;"
                    )
            elif canonical_skill in secondary_canonical:
                for alias in aliases:
                    if secondary_color == "red":
                        alias_to_style[alias] = (
                            "background-color: rgba(239, 85, 59, 0.28); "
                            "border-bottom: 2px solid #EF553B; "
                            "padding: 1px 4px; border-radius: 4px; font-weight: 600;"
                        )
                    else:
                        alias_to_style[alias] = (
                            "background-color: rgba(99, 110, 250, 0.25); "
                            "border-bottom: 2px solid #636EFA; "
                            "padding: 1px 4px; border-radius: 4px; font-weight: 500;"
                        )

    if not alias_to_style:
        return escaped_text.replace("\n", "<br>")

    sorted_aliases = sorted(alias_to_style.keys(), key=len, reverse=True)
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(html.escape(a)) for a in sorted_aliases) + r')\b',
        flags=re.IGNORECASE
    )

    def _replace(match):
        matched_word = match.group(0)
        style = alias_to_style.get(matched_word.lower(), "")
        return f'<mark style="color: inherit; {style}">{matched_word}</mark>'

    highlighted = pattern.sub(_replace, escaped_text)
    return highlighted.replace("\n", "<br>")

def estimate_experience_signals(resume_text: str, jd_text: str) -> dict:
    year_matches = re.findall(r'(\d{1,2})\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:[\w\s,&/-]{0,30}?)(?:experience|exp)', resume_text.lower())
    years_found = max([int(y) for y in year_matches], default=0)

    jd_year_matches = re.findall(r'(\d{1,2})\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:[\w\s,&/-]{0,30}?)(?:experience|exp)', jd_text.lower())
    jd_years_req = max([int(y) for y in jd_year_matches], default=0)

    return {
        "resume_years": years_found,
        "jd_years": jd_years_req,
        "exp_gap": (years_found - jd_years_req) if (jd_years_req > 0 and years_found > 0) else 0
    }

def build_interview_prep(
    role_title: str,
    unique_matched: list[str],
    unique_missing: list[str],
    axis_details: dict,
    top_axis: str,
    low_axis: str,
    exp_info: dict
) -> dict:
    role_label = (role_title or "this role").strip()
    focus_points = []
    questions_to_ask = []

    strong_axes = [
        (cat, info) for cat, info in axis_details.items()
        if info["matched"] or info["score"] >= 75.0
    ]
    if strong_axes:
        top_cat, top_info = max(strong_axes, key=lambda item: (len(item[1]["matched"]), item[1]["score"]))
        matched_on_top = [s.title() for s in top_info["matched"]]
        if matched_on_top:
            focus_points.append(
                f"**{top_cat} ({top_info['score']}% match):** Lead with your verified JD matches in this parse—**{', '.join(matched_on_top)}**—and walk through how you built repeatable reports, models, or workflows that delivered measurable results."
            )
        else:
            resume_on_top = [s.title() for s in top_info["resume_has"][:4]]
            focus_points.append(
                f"**{top_cat} ({top_info['score']}% readiness):** Highlight your resume depth in **{', '.join(resume_on_top) if resume_on_top else top_cat}** and connect it directly to the operational problems described in this job posting."
            )

    other_matched = [s.title() for s in unique_matched if s not in axis_details.get(top_axis, {}).get("matched", [])]
    if other_matched:
        focus_points.append(
            f"**Cross-Domain Matched Proof ({', '.join(other_matched[:5])}):** Prepare one crisp STAR example per matched skill showing data accuracy, stakeholder adoption, and business or labor cost impact."
        )
    elif unique_matched:
        focus_points.append(
            f"**Core Matched Capabilities ({', '.join([s.title() for s in unique_matched[:5]])}):** Frame each matched capability around how quickly it moves an insight from raw data to an operational decision."
        )

    gap_axes = [(cat, info) for cat, info in axis_details.items() if info["missing"]]
    if unique_missing:
        missing_titles = [s.title() for s in unique_missing]
        gap_cat_names = [cat for cat, _ in gap_axes]
        focus_points.append(
            f"**Bridge Unmatched JD Skills ({', '.join(missing_titles[:5])}):** For skills flagged missing in **{', '.join(gap_cat_names[:3]) or low_axis}**, point to adjacent tools, analytical methods, or domain work you already execute and explain how you ramp up fast."
        )
    else:
        focus_points.append(
            "**Zero Missing JD Skills Flagged:** Every tracked skill in this job description matched your resume—use the interview to go deep on architecture, edge cases, and executive communication rather than basic tool proficiency."
        )

    if exp_info["jd_years"] > 0 and exp_info["resume_years"] > 0:
        if exp_info["exp_gap"] < 0:
            focus_points.append(
                f"**Tenure Framing (~{exp_info['resume_years']} yrs parsed vs. {exp_info['jd_years']}+ yrs in JD):** Emphasize end-to-end ownership, cross-functional scope, and quantified cost/efficiency wins to offset the {abs(exp_info['exp_gap'])}-year nominal gap."
            )
        else:
            focus_points.append(
                f"**Tenure Advantage (~{exp_info['resume_years']} yrs parsed vs. {exp_info['jd_years']}+ yrs in JD):** Your experience meets or exceeds the posting's tenure bar—emphasize mentoring, process standardization, and strategic decision support."
            )

    questions_to_ask.append(
        f"**Immediate Priorities for {role_label}:** Given the focus on **{low_axis}** and the core deliverables in this posting, what is the single highest-priority problem you need solved in the first 60–90 days?"
    )

    if unique_missing:
        top_gaps_q = ", ".join([s.title() for s in unique_missing[:3]])
        questions_to_ask.append(
            f"**Role Expectations Around {top_gaps_q}:** How central are **{top_gaps_q}** to day-to-day execution on this team, and where is there flexibility to leverage adjacent analytics, BI, and operational workflows?"
        )

    if unique_matched:
        top_matched_q = ", ".join([s.title() for s in unique_matched[:3]])
        questions_to_ask.append(
            f"**Applying {top_matched_q} in Production:** How does the team currently use **{top_matched_q}** in its reporting or decision pipeline, and what data-quality or workflow bottlenecks would you want improved first?"
        )

    questions_to_ask.append(
        f"**Success Metrics & Stakeholder Alignment:** How is success measured for this role over the first six months across **{top_axis}** and **{low_axis}**, and which teams partner most closely with the analyst?"
    )

    questions_to_ask.append(
        "**Team & Tooling Roadmap:** What tools, data sources, or reporting processes are actively evolving over the next year, and how does the team support analysts in adopting new workflows?"
    )

    return {
        "focus_points": focus_points[:5],
        "questions_to_ask": questions_to_ask[:5]
    }

def analyze_candidate_fit(
    resume_text: str,
    jd_text: str,
    benchmark_mean: float = 65.0,
    benchmark_std: float = 12.0,
    role_title: str = "Data and Business Analyst"
) -> dict:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    raw_cosine = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    contextual_score = min(100.0, round(raw_cosine * 210, 1))

    res_skills = extract_skills_by_category(resume_text)
    jd_skills = extract_skills_by_category(jd_text)

    category_breakdown = []
    axis_details = {}
    total_jd_skills = 0
    total_matched_skills = 0
    all_missing = []
    all_matched = []
    all_resume_skills = []

    for cat in COMPETENCY_TAXONOMY.keys():
        req = jd_skills[cat]
        has = res_skills[cat]
        matched = sorted(list(req.intersection(has)))
        missing = sorted(list(req - has))

        total_jd_skills += len(req)
        total_matched_skills += len(matched)
        all_missing.extend(missing)
        all_matched.extend(matched)
        all_resume_skills.extend(list(has))

        if len(req) > 0:
            cat_score = round((len(matched) / len(req)) * 100, 1)
        else:
            axis_total = len(COMPETENCY_TAXONOMY[cat])
            cat_score = round(min(100.0, (len(has) / max(1, axis_total)) * 140), 1)

        axis_details[cat] = {
            "required": sorted(list(req)),
            "resume_has": sorted(list(has)),
            "matched": matched,
            "missing": missing,
            "score": cat_score
        }

        category_breakdown.append({
            "Category": cat,
            "JD Required": len(req),
            "Matched": len(matched),
            "Missing": len(missing),
            "Resume Breadth": len(has),
            "Category Match (%)": cat_score
        })

    skill_match_score = (total_matched_skills / total_jd_skills * 100) if total_jd_skills > 0 else contextual_score

    hr_score = next((r["Category Match (%)"] for r in category_breakdown if r["Category"] == "HR & People Analytics"), 70.0)
    ops_score = next((r["Category Match (%)"] for r in category_breakdown if r["Category"] == "Workforce & Labor Ops"), 70.0)
    lead_score = next((r["Category Match (%)"] for r in category_breakdown if r["Category"] == "Leadership & Strategy"), 70.0)
    people_analytics_index = round((0.40 * hr_score) + (0.35 * ops_score) + (0.25 * lead_score), 1)

    exp_info = estimate_experience_signals(resume_text, jd_text)
    composite_score = round((0.50 * skill_match_score) + (0.30 * contextual_score) + (0.20 * people_analytics_index), 1)

    z_score = (composite_score - benchmark_mean) / benchmark_std
    percentile = round(norm.cdf(z_score) * 100, 1)

    if percentile >= 85:
        tier = "Top Performer Tier (+1σ to +2σ)"
        ramp_up = "Fast (2–4 weeks to full productivity)"
        retention_prob = "High (88%–94%)"
        attrition_risk = "Low"
    elif percentile >= 50:
        tier = "Above Average / Core Contributor (Mean to +1σ)"
        ramp_up = "Standard (4–8 weeks to full productivity)"
        retention_prob = "Moderate-High (75%–87%)"
        attrition_risk = "Low-Moderate"
    elif percentile >= 25:
        tier = "Developing / Coaching Required (-1σ to Mean)"
        ramp_up = "Extended (8–12 weeks with structured onboarding)"
        retention_prob = "Moderate (60%–74%)"
        attrition_risk = "Moderate"
    else:
        tier = "High Risk / Significant Skill Gap (Below -1σ)"
        ramp_up = "Slow (12+ weeks; requires foundational training)"
        retention_prob = "Low (<60%)"
        attrition_risk = "Elevated"

    cat_df = pd.DataFrame(category_breakdown)
    top_axis_row = cat_df.sort_values("Category Match (%)", ascending=False).iloc[0]
    low_axis_row = cat_df.sort_values("Category Match (%)", ascending=True).iloc[0]

    unique_matched = sorted(set(all_matched))
    unique_missing = sorted(set(all_missing))
    bonus_resume_skills = sorted(set(all_resume_skills) - set(unique_matched))

    resume_highlighted_html = build_highlighted_html(
        text=resume_text,
        green_canonical=unique_matched,
        secondary_canonical=bonus_resume_skills,
        secondary_color="blue"
    )
    jd_highlighted_html = build_highlighted_html(
        text=jd_text,
        green_canonical=unique_matched,
        secondary_canonical=unique_missing,
        secondary_color="red"
    )

    # Percentile-driven Strong Match & Apply Verdict
    if percentile >= 75.0:
        match_strength_label = "STRONG MATCH TO APPLY"
        apply_decision = "STRONG MATCH — RECOMMENDED TO APPLY"
        apply_badge_type = "success"
        is_good_match = True
        apply_summary = (
            f"Your resume is a STRONG MATCH at the {percentile}th percentile (Z = {z_score:+.2f}, composite {composite_score}%), "
            f"placing you in the top quartile of the pool (μ={benchmark_mean}, σ={benchmark_std}). Apply promptly and prep for your interview below."
        )
    elif percentile >= 50.0:
        match_strength_label = "MODERATE MATCH — TAILOR & APPLY"
        apply_decision = "MODERATE MATCH — TAILOR RESUME THEN APPLY"
        apply_badge_type = "warning"
        is_good_match = True
        apply_summary = (
            f"Your resume is a MODERATE MATCH at the {percentile}th percentile (Z = {z_score:+.2f}, composite {composite_score}%), "
            f"above the pool average (μ={benchmark_mean}). Tailor your resume to match JD wording before applying."
        )
    else:
        match_strength_label = "NOT A STRONG MATCH TO APPLY"
        apply_decision = "NOT A STRONG MATCH — DO NOT APPLY YET"
        apply_badge_type = "error"
        is_good_match = False
        apply_summary = (
            f"Your resume is NOT A STRONG MATCH at the {percentile}th percentile (Z = {z_score:+.2f}, composite {composite_score}%), "
            f"below the pool median (μ={benchmark_mean}). Focus on higher-percentile roles or strengthen missing skills first."
        )

    interview_prep = build_interview_prep(
        role_title=role_title,
        unique_matched=unique_matched,
        unique_missing=unique_missing,
        axis_details=axis_details,
        top_axis=top_axis_row["Category"],
        low_axis=low_axis_row["Category"],
        exp_info=exp_info
    )

    insights = [
        f"Percentile Verdict ({match_strength_label}): You rank at the {percentile}th percentile (Z = {z_score:+.2f}, composite {composite_score}%) against a μ={benchmark_mean}, σ={benchmark_std} pool.",
        f"Competency Radar Balance: Strongest domain is {top_axis_row['Category']} ({top_axis_row['Category Match (%)']}%), while {low_axis_row['Category']} ({low_axis_row['Category Match (%)']}%) is the primary area to watch.",
        f"HR & People Analytics Signal: People Analytics Index is {people_analytics_index}% (HR/People: {hr_score}%, Workforce/Labor Ops: {ops_score}%, Leadership: {lead_score}%), projecting {ramp_up.lower()}.",
        f"Keyword & Narrative Alignment: Matched {len(unique_matched)} JD skill areas and achieved a {contextual_score}% contextual phrasing score against {len(unique_missing)} unmatched JD skill area(s)."
    ]

    if exp_info["jd_years"] > 0 and exp_info["resume_years"] > 0:
        if exp_info["exp_gap"] < 0:
            insights.append(
                f"Tenure & Experience Delta: Resume indicates ~{exp_info['resume_years']} years of experience vs. {exp_info['jd_years']}+ years referenced in the job description ({exp_info['exp_gap']} yrs delta)."
            )
        else:
            insights.append(
                f"Tenure & Experience Delta: Resume (~{exp_info['resume_years']} yrs) meets or exceeds the {exp_info['jd_years']}+ yrs benchmark ({exp_info['exp_gap']:+d} yrs delta)."
            )

    recommendations = []
    if percentile >= 75.0:
        recommendations.append(f"Strong {percentile}th-percentile match—submit your application right away and lead with your top matched tools and quantified outcomes.")
    elif percentile >= 50.0:
        recommendations.append(f"Moderate {percentile}th-percentile match—mirror key phrases from the job description in your summary and top bullets to push into the 75th+ percentile range.")
    else:
        recommendations.append(f"Below-median {percentile}th-percentile match—skip this posting for now and target roles where your core skills put you at the 75th percentile or higher.")

    if unique_matched:
        top_m = ", ".join([s.title() for s in unique_matched[:5]])
        recommendations.append(f"Highlight quantified outcomes alongside your verified strengths: {top_m}.")

    if unique_missing:
        top_x = ", ".join([s.title() for s in unique_missing[:5]])
        recommendations.append(f"Address missing JD keywords by surfacing adjacent projects or coursework for: {top_x}.")
    else:
        recommendations.append("All detected job description skills are matched—differentiate yourself with concise before/after operational impact stories.")

    if contextual_score < skill_match_score - 12:
        recommendations.append("Boost ATS and recruiter alignment by incorporating exact action verbs and KPI terms from the job posting.")

    return {
        "match_strength_label": match_strength_label,
        "apply_decision": apply_decision,
        "apply_badge_type": apply_badge_type,
        "apply_summary": apply_summary,
        "is_good_match": is_good_match,
        "interview_prep": interview_prep,
        "resume_highlighted_html": resume_highlighted_html,
        "jd_highlighted_html": jd_highlighted_html,
        "composite_score": composite_score,
        "skill_match_score": round(skill_match_score, 1),
        "contextual_score": contextual_score,
        "people_analytics_index": people_analytics_index,
        "z_score": round(z_score, 2),
        "percentile": percentile,
        "tier": tier,
        "ramp_up": ramp_up,
        "retention_prob": retention_prob,
        "attrition_risk": attrition_risk,
        "exp_info": exp_info,
        "matched_skills": unique_matched,
        "missing_skills": unique_missing,
        "axis_details": axis_details,
        "category_df": cat_df,
        "insights": insights,
        "recommendations": recommendations
    }

def build_indeed_easy_apply_links(
    role_query: str,
    location: str = "",
    remote_only: bool = False,
    matched_skills: list | None = None,
    missing_skills: list | None = None
) -> list[dict]:
    base_role = (role_query or "Data Analyst").strip()
    loc = (location or "").strip()
    matched = [s.title() for s in (matched_skills or [])]
    missing = [s.title() for s in (missing_skills or [])]

    queries = [
        {
            "Label": "Primary Role Match (Easy Apply)",
            "Query": base_role,
            "Reason": f"Targets '{base_role}' roles on Indeed with Easy Apply enabled."
        }
    ]

    if matched:
        top_matched = " ".join(matched[:2])
        queries.append({
            "Label": f"Strength-Aligned ({top_matched})",
            "Query": f"{base_role} {top_matched}".strip(),
            "Reason": f"Leverages your matched strengths ({', '.join(matched[:3])}) for higher initial fit."
        })

    if missing:
        top_gap = missing[0]
        queries.append({
            "Label": f"Skill-Stretch Role (+{top_gap})",
            "Query": f"{base_role} {top_gap}".strip(),
            "Reason": f"Explores roles emphasizing '{top_gap}' where you can highlight adjacent experience."
        })

    queries.append({
        "Label": "HR & Workforce Analytics Focus",
        "Query": "People Analytics HR Analyst Workforce",
        "Reason": "Targets HR, workforce planning, scheduling, and people analytics roles."
    })

    links = []
    for item in queries:
        params = [f"q={quote_plus(item['Query'])}", "sritch=1"]
        if loc:
            params.append(f"l={quote_plus(loc)}")
        if remote_only:
            params.append("sc=0kf%3Aattr(DSQF7)%3B")

        url = "https://www.indeed.com/jobs?" + "&".join(params)
        links.append({
            "Link Title": item["Label"],
            "Search Terms": item["Query"],
            "Location Filter": "Remote" if remote_only and not loc else (loc or "All Locations"),
            "Why This Link": item["Reason"],
            "Indeed Easy Apply URL": url
        })

    return links
