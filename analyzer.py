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
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

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
    top_axis: str,
    hr_score: float,
    ops_score: float,
    exp_info: dict
) -> dict:
    """Generates personalized interview talking points and questions to ask the interviewer."""
    role_label = (role_title or "this role").strip()
    matched_titles = [s.title() for s in unique_matched]
    missing_titles = [s.title() for s in unique_missing]

    focus_points = []

    if matched_titles:
        core_tools = ", ".join(matched_titles[:5])
        focus_points.append(
            f"**Lead with Core Stack Proof ({core_tools}):** Walk through a concrete before-and-after example where you turned raw operational, payroll, or scheduling records into automated dashboards and decision-ready reports."
        )

    focus_points.append(
        f"**Anchor on Your Strongest Domain ({top_axis}):** Highlight measurable business impact—such as reducing operating/labor costs by 15%, cutting manual spreadsheet work, and improving staffing/forecast accuracy."
    )

    if hr_score >= 60 or ops_score >= 60:
        focus_points.append(
            "**Connect Workforce Data to Financial Outcomes:** Emphasize how you bridge HR/payroll compliance, timekeeping, and headcount utilization directly to revenue protection and margin control."
        )

    if missing_titles:
        gaps_str = ", ".join(missing_titles[:4])
        focus_points.append(
            f"**Prepare a Bridge Story for Skill Gaps ({gaps_str}):** Acknowledge where your direct title exposure is lighter, then pivot immediately to adjacent analytical workflows and fast self-directed ramp-up."
        )
    else:
        focus_points.append(
            "**Differentiate with Executive Storytelling:** Since all core JD skills align, focus on how you translate complex SQL/Python/BI outputs into clear recommendations for non-technical leaders."
        )

    if exp_info["jd_years"] > 0 and exp_info["resume_years"] > 0 and exp_info["exp_gap"] < 0:
        focus_points.append(
            f"**Address Tenure Delta ({exp_info['resume_years']} yrs vs. {exp_info['jd_years']}+ yrs):** Frame your multi-industry operations and analytics background as compressed, high-ownership experience across full reporting lifecycles."
        )

    questions_to_ask = [
        f"**90-Day Impact & Priorities:** What are the top one or two reporting bottlenecks or operational decisions you need the {role_label} to solve in the first 90 days?",
        "**Data Stack & Source Maturity:** Where does the team's data currently live (HRIS, payroll, ERP, spreadsheets, or SQL warehouse), and how much time is spent on data cleaning vs. analysis?",
        "**Key Performance Metrics:** Which metrics matter most to leadership right now—labor cost variance, turnover/retention, forecast accuracy, or process turnaround time?",
        "**Cross-Functional Stakeholders:** Who are the primary internal consumers of these dashboards and insights (HR, Finance, Operations, or Executive Leadership), and how frequently do they review them?"
    ]

    if missing_titles:
        questions_to_ask.append(
            f"**Tooling & Workflow Depth:** I noticed {missing_titles[0]} mentioned in the job description—how heavily is that used day-to-day versus your core BI and spreadsheet tools?"
        )

    return {
        "focus_points": focus_points,
        "questions_to_ask": questions_to_ask
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

    for cat in COMPETENCY_TAXONOMY.keys():
        req = jd_skills[cat]
        has = res_skills[cat]
        matched = sorted(list(req.intersection(has)))
        missing = sorted(list(req - has))

        total_jd_skills += len(req)
        total_matched_skills += len(matched)
        all_missing.extend(missing)
        all_matched.extend(matched)

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

    severe_exp_gap = (exp_info["jd_years"] > 0 and exp_info["resume_years"] > 0 and exp_info["exp_gap"] <= -3)
    is_good_match = False

    if composite_score >= 68.0 and len(unique_missing) <= 4 and not severe_exp_gap:
        apply_decision = "RECOMMENDED TO APPLY"
        apply_badge_type = "success"
        is_good_match = True
        apply_summary = (
            f"Strong statistical and competency alignment ({composite_score}% composite, {percentile}th percentile). "
            f"Apply promptly and use the Interview Prep Focus Points below to prepare for your invite."
        )
    elif composite_score >= 48.0 and not severe_exp_gap:
        apply_decision = "CONDITIONAL APPLY (TAILOR FIRST)"
        apply_badge_type = "warning"
        is_good_match = True
        apply_summary = (
            f"Solid baseline alignment ({composite_score}% composite, {percentile}th percentile). "
            f"Tailor your top bullets to mirror the JD, then use the Interview Prep Focus Points below."
        )
    else:
        apply_decision = "NOT RECOMMENDED TO APPLY NOW"
        apply_badge_type = "error"
        is_good_match = False
        apply_summary = (
            f"Low current alignment ({composite_score}% composite, {percentile}th percentile) or wide experience gap. "
            f"Prioritize higher-fit roles from the Indeed Easy Apply links below."
        )

    interview_prep = build_interview_prep(
        role_title=role_title,
        unique_matched=unique_matched,
        unique_missing=unique_missing,
        top_axis=top_axis_row["Category"],
        hr_score=hr_score,
        ops_score=ops_score,
        exp_info=exp_info
    )

    insights = [
        f"Overall Fit & Distribution: {composite_score}% composite score places your profile at the {percentile}th percentile (Z = {z_score:+.2f}) relative to μ={benchmark_mean}, σ={benchmark_std}.",
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
    if apply_decision == "RECOMMENDED TO APPLY":
        recommendations.append("Submit your application promptly while the posting is active; lead your summary and top 3 bullets with your strongest matched tools and quantified outcomes.")
    elif apply_decision == "CONDITIONAL APPLY (TAILOR FIRST)":
        recommendations.append("Tailor your resume summary and top experience bullets to mirror the job description's exact phrasing before applying.")
    else:
        recommendations.append("Skip low-ROI applications here and focus on the pre-filtered Indeed Easy Apply searches below that match your core analytics stack.")

    if unique_matched:
        top_m = ", ".join([s.title() for s in unique_matched[:5]])
        recommendations.append(f"Highlight quantified outcomes (15% cost reduction, automated reporting hours saved, forecast accuracy) alongside your verified strengths: {top_m}.")

    if unique_missing:
        top_x = ", ".join([s.title() for s in unique_missing[:5]])
        recommendations.append(f"Address missing JD keywords by surfacing adjacent projects or coursework for: {top_x}.")
    else:
        recommendations.append("All detected job description skills are matched—differentiate yourself with concise before/after operational impact stories.")

    if contextual_score < skill_match_score - 12:
        recommendations.append("Boost ATS and recruiter alignment by incorporating exact action verbs and KPI terms from the job posting.")

    return {
        "apply_decision": apply_decision,
        "apply_badge_type": apply_badge_type,
        "apply_summary": apply_summary,
        "is_good_match": is_good_match,
        "interview_prep": interview_prep,
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
