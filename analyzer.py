import html
import re
from urllib.parse import quote_plus
import numpy as np
import pandas as pd
from pypdf import PdfReader
from docx import Document
from fpdf import FPDF
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

def build_company_brief(jd_text: str, company_name: str, role_title: str) -> dict:
    jd_lower = jd_text.lower()
    detected_name = (company_name or "").strip()

    if not detected_name:
        patterns = [
            r'(?:about\s+us|who\s+we\s+are|join\s+)[:\-\s]*([A-Z][A-Za-z0-9&.,\s]{2,40}?)(?:\s+is\b|\s+seeks\b|\s+offers\b|\.)',
            r'([A-Z][A-Za-z0-9&.-]+(?:\s+[A-Z][A-Za-z0-9&.-]+){0,3})\s+is\s+(?:a\s+leading|an\s+innovative|a\s+fast-growing|looking\s+for)'
        ]
        for pat in patterns:
            m = re.search(pat, jd_text)
            if m:
                candidate = m.group(1).strip(" .,-")
                if len(candidate.split()) <= 5 and candidate.lower() not in {"the role", "this position", "we", "our team"}:
                    detected_name = candidate
                    break

    company_label = detected_name if detected_name else "the hiring organization"

    about_match = re.search(
        r'(?:about\s+(?:the\s+company|us|organization)|who\s+we\s+are)[:\s\-]*([^\n]+(?:\n[^\n]+){0,2})',
        jd_text,
        flags=re.IGNORECASE
    )
    jd_excerpt = re.sub(r'\s+', ' ', about_match.group(1)).strip() if about_match else ""

    themes = []
    if any(k in jd_lower for k in ["hospitality", "hotel", "resort", "guest", "event", "convention"]):
        themes.append("hospitality, guest/event operations, or high-touch service delivery")
    if any(k in jd_lower for k in ["payroll", "hris", "workforce", "headcount", "scheduling", "human resources", "people"]):
        themes.append("workforce management, scheduling/payroll accuracy, and people analytics")
    if any(k in jd_lower for k in ["sql", "power bi", "tableau", "python", "dashboard", "forecast", "analytics", "bi"]):
        themes.append("data-driven reporting, BI visualization, and operational forecasting")
    if any(k in jd_lower for k in ["compliance", "audit", "process improvement", "efficiency", "cost", "budget"]):
        themes.append("cost discipline, compliance rigor, and repeatable process improvement")

    theme_summary = (
        f"Based on the job description, {company_label} is hiring for **{role_title or 'this role'}** with emphasis on "
        + (", ".join(themes[:-1]) + f", and {themes[-1]}" if len(themes) > 1 else (themes[0] if themes else "structured analytics and cross-functional execution"))
        + "."
    )

    research_checks = [
        f"Review {company_label}'s recent news, leadership priorities, and core products/services to tie your analytics recommendations directly to their revenue and cost goals.",
        f"Look up the team structure on LinkedIn to see how {role_title or 'this role'} partners with Operations, HR/People, Finance, and leadership.",
        "Note specific metrics, software platforms, or compliance rules mentioned in the posting so you can reference them naturally in conversation."
    ]

    return {
        "company_label": company_label,
        "theme_summary": theme_summary,
        "jd_excerpt": jd_excerpt,
        "research_checks": research_checks
    }

def build_tell_me_about_yourself(
    role_title: str,
    company_label: str,
    unique_matched: list[str],
    top_axis: str,
    percentile: float,
    exp_info: dict
) -> dict:
    role_label = (role_title or "this analyst role").strip()
    matched_top = [s.title() for s in unique_matched[:4]]
    tools_phrase = ", ".join(matched_top) if matched_top else "Excel, SQL, Power BI, and Python"

    yrs_phrase = (
        f"with roughly {exp_info['resume_years']} years of hands-on experience"
        if exp_info["resume_years"] > 0
        else "with extensive hands-on experience"
    )

    pitch_30_60 = (
        f"\"I'm a data and operations analyst {yrs_phrase} turning workforce, scheduling, payroll, and operational records "
        f"into clear reporting and decision support using {tools_phrase}. In my recent roles, I've built automated tracking "
        f"and forecasting workflows that cut manual reporting effort, improved labor and compliance visibility, and helped reduce "
        f"operating expenses by 15%. Because my strongest alignment for this {role_label} position is in {top_axis} "
        f"({percentile}th percentile fit), I'm excited about the opportunity to bring structured analysis and practical execution "
        f"to {company_label}.\""
    )

    key_pillars = [
        f"**Opening Hook:** Start with your core identity—an analytical operator who pairs {tools_phrase} with real-world workforce and operational context.",
        "**Proof of Impact:** Cite concrete outcomes—cutting operating expenses by 15%, replacing error-prone manual tracking, and delivering forecasts leaders can act on quickly.",
        f"**Why This Role & Company ({company_label}):** Connect your {top_axis} strength directly to the core responsibilities of {role_label}."
    ]

    return {
        "pitch_script": pitch_30_60,
        "key_pillars": key_pillars
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

def _safe_pdf_text(txt: str) -> str:
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2022": "-", "\u2192": "->",
        "\u03c3": "sigma", "\u03bc": "mu"
    }
    for k, v in replacements.items():
        txt = txt.replace(k, v)
    ascii_safe = txt.encode("latin-1", errors="ignore").decode("latin-1")
    # Break any unbroken token longer than 45 chars so fpdf2 never fails on width
    return re.sub(r'(\S{45})', r'\1 ', ascii_safe)

def build_prescriptive_smart_resume(
    resume_text: str,
    role_title: str,
    company_label: str,
    unique_matched: list[str],
    axis_details: dict,
    top_axis: str,
    use_smart_goals: bool = True
) -> dict:
    lines = [l.strip() for l in resume_text.splitlines() if l.strip()]
    candidate_name = lines[0][:80] if lines else "MARK ANTHONY RECIO, MBA"

    contact_line = "Las Vegas, NV | 702-695-8470 | mark.recio@marec.site | markreciopro.github.io/Portfolio"
    for l in lines[1:4]:
        if "@" in l or "linkedin" in l.lower() or "702" in l:
            contact_line = l[:120]
            break

    role_header = (role_title or "DATA & BUSINESS ANALYST").upper()[:70]
    matched_titles = [s.title() for s in unique_matched]
    top_tools = ", ".join(matched_titles[:6]) if matched_titles else "Excel, SQL, Power BI, Python, R, Tableau"

    if use_smart_goals:
        tailored_summary = (
            f"Results-driven {role_title or 'Data & Business Analyst'} with 8+ years of operational, workforce, and reporting "
            f"analytics experience. Proven track record of delivering measurable business outcomes-including a 15% reduction in "
            f"operating expenses, automated reporting pipelines using {top_tools}, and high-accuracy staffing forecasts-aligned "
            f"to drive immediate 90-day impact at {company_label}."
        )
    else:
        tailored_summary = (
            f"Analytical and detail-oriented {role_title or 'Data & Business Analyst'} with 8+ years of experience turning "
            f"payroll, timekeeping, scheduling, and operational datasets into clear dashboards and decision support using "
            f"{top_tools}. Strongest alignment in {top_axis} to support {company_label}'s reporting and operational goals."
        )

    data_skills = sorted(set(axis_details.get("Data & BI Tools", {}).get("resume_has", []) + axis_details.get("Analytics & Forecasting", {}).get("resume_has", [])))
    hr_ops_skills = sorted(set(axis_details.get("HR & People Analytics", {}).get("resume_has", []) + axis_details.get("Workforce & Labor Ops", {}).get("resume_has", [])))
    proc_lead_skills = sorted(set(axis_details.get("Process & Compliance", {}).get("resume_has", []) + axis_details.get("Leadership & Strategy", {}).get("resume_has", [])))

    def _prioritize(skill_list, fallback):
        if not skill_list:
            return fallback
        matched_first = [s.title() for s in skill_list if s in unique_matched]
        others = [s.title() for s in skill_list if s not in unique_matched]
        return ", ".join(matched_first + others)

    skills_sections = {
        "Core Matched & Technical Stack": _prioritize(data_skills, "Excel (Advanced), Power BI, Tableau, SQL, Python, R, Data Cleaning, Forecasting"),
        "HR, Workforce & Operations": _prioritize(hr_ops_skills, "Workforce Planning, Payroll & Timekeeping, Scheduling, Labor Cost Control, Operations"),
        "Process, Compliance & Strategy": _prioritize(proc_lead_skills, "Process Improvement, Work-Rule Compliance, KPI Reporting, Stakeholder Presentations")
    }

    if use_smart_goals:
        exp_role_1_bullets = [
            f"[Specific & Measurable] Consolidated multi-source payroll, recruiting, and operational datasets using {top_tools}, eliminating manual reconciliation bottlenecks and improving weekly reporting turnaround by 40%.",
            f"[Relevant & Time-Bound] Designed and maintained executive KPI dashboards tracking labor cost, utilization, turnover, and staffing demand across active client accounts to support weekly scheduling decisions.",
            "[Measurable & Achievable] Built predictive staffing and demand forecasts for recurring operational cycles, aligning headcount allocation with fluctuating workload targets.",
            "[Specific & Relevant] Audited AI scheduling and timekeeping outputs against work-rule compliance standards, delivering concise executive findings that prevented payroll discrepancies."
        ]
        exp_role_2_bullets = [
            "[Measurable & Time-Bound] Analyzed crew scheduling, timekeeping, and labor utilization across multi-site Las Vegas operations (2018-2023), driving process improvements that reduced operating expenses by 15%.",
            "[Specific & Achievable] Engineered standardized Excel scheduling, payroll, and compliance trackers for teams of 10-50+ staff, cutting manual administrative preparation time each pay period.",
            "[Relevant & Measurable] Supervised event crews and coordinated end-to-end logistics across high-volume operations while maintaining 100% recordkeeping and regulatory compliance."
        ]
    else:
        exp_role_1_bullets = [
            f"Combine payroll, recruiting, and operations data into consistent reports and dashboards using {top_tools}.",
            "Build and maintain dashboards tracking labor cost, productivity, turnover, and recruiting performance.",
            "Prepare demand and staffing forecasts that support operational scheduling decisions and present findings in clear reports.",
            "Audit automated scheduling tools for accuracy and compliance with established work rules."
        ]
        exp_role_2_bullets = [
            "Reviewed scheduling and labor data for service and event crews across Las Vegas operations, reducing operating expenses by 15%.",
            "Built Excel scheduling, payroll, timekeeping, and compliance trackers for teams of 10-50 staff, cutting manual administrative effort.",
            "Supervised event crews, coordinated logistics, and managed day-to-day operational administration."
        ]

    education_certs = [
        "MBA, Keller Graduate School of Management | Graduate Certificate in Strategic Innovation & Change, University of Denver | B.S., Technical Management, DeVry University",
        "Certifications: Microsoft Power BI Data Analyst Associate | Tableau Business Intelligence Analyst | Google Data Analytics | IBM Data Science | Business Analytics (Univ. of Illinois) | HRCI Associate | Nevada HR & Payroll Certificate"
    ]

    pdf = FPDF(format="letter")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(15, 76, 92)
    pdf.multi_cell(usable_w, 7, _safe_pdf_text(candidate_name))

    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(usable_w, 5.5, _safe_pdf_text(f"TARGET ROLE: {role_header} | {company_label.upper()[:50]}"))

    pdf.set_font("Helvetica", "", 8.8)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(usable_w, 5, _safe_pdf_text(contact_line))
    pdf.ln(2)

    def _section_title(title: str):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(15, 76, 92)
        pdf.multi_cell(usable_w, 5.5, _safe_pdf_text(title))
        y_pos = pdf.get_y()
        pdf.set_draw_color(15, 76, 92)
        pdf.line(pdf.l_margin, y_pos, pdf.w - pdf.r_margin, y_pos)
        pdf.ln(1.5)

    _section_title("TARGETED PROFESSIONAL SUMMARY")
    pdf.set_font("Helvetica", "", 9.2)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(usable_w, 4.6, _safe_pdf_text(tailored_summary))
    pdf.ln(2)

    _section_title("ROLE-ALIGNED SKILLS & ANALYTICS STACK")
    for s_label, s_vals in skills_sections.items():
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(usable_w, 4.5, _safe_pdf_text(f"{s_label}: {s_vals}"))
    pdf.ln(2)

    exp_heading = "EXPERIENCE (SMART-GOAL IMPACT ALIGNMENT)" if use_smart_goals else "RELEVANT ANALYTICS & OPERATIONS EXPERIENCE"
    _section_title(exp_heading)

    pdf.set_font("Helvetica", "B", 9.4)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(usable_w, 5, _safe_pdf_text("Contract Data & Workforce Analyst | Synergy Staffing Solutions (via MAREC Insights) - Las Vegas, NV (Aug 2023 - Present)"))
    pdf.set_font("Helvetica", "", 9)
    for b in exp_role_1_bullets:
        pdf.multi_cell(usable_w, 4.5, _safe_pdf_text(f"  - {b}"))
    pdf.ln(1.5)

    pdf.set_font("Helvetica", "B", 9.4)
    pdf.multi_cell(usable_w, 5, _safe_pdf_text("Operations & Workforce Coordinator | Bella Brite Services - Las Vegas, NV (Oct 2018 - Aug 2023)"))
    pdf.set_font("Helvetica", "", 9)
    for b in exp_role_2_bullets:
        pdf.multi_cell(usable_w, 4.5, _safe_pdf_text(f"  - {b}"))
    pdf.ln(2)

    _section_title("EDUCATION & INDUSTRY CERTIFICATIONS")
    pdf.set_font("Helvetica", "", 9)
    for ec in education_certs:
        pdf.multi_cell(usable_w, 4.5, _safe_pdf_text(f"  - {ec}"))

    raw_out = pdf.output()
    pdf_bytes = bytes(raw_out) if isinstance(raw_out, (bytes, bytearray)) else raw_out.encode("latin-1")

    preview_md = (
        f"### {candidate_name}\n"
        f"**Target Role:** {role_title} | **Organization:** {company_label}\n\n"
        f"#### Targeted Summary\n{tailored_summary}\n\n"
        f"#### Prioritized Skills\n"
        + "\n".join([f"- **{k}:** {v}" for k, v in skills_sections.items()])
        + "\n\n#### Experience Highlights"
        + (" *(SMART Goal Format)*\n" if use_smart_goals else "\n")
        + "\n".join([f"- {b}" for b in exp_role_1_bullets + exp_role_2_bullets])
    )

    return {
        "pdf_bytes": pdf_bytes,
        "preview_md": preview_md,
        "smart_enabled": use_smart_goals
    }

def analyze_candidate_fit(
    resume_text: str,
    jd_text: str,
    benchmark_mean: float = 65.0,
    benchmark_std: float = 12.0,
    role_title: str = "Data and Business Analyst",
    company_name: str = "",
    use_smart_goals: bool = True
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

    if percentile >= 75.0:
        match_strength_label = "STRONG MATCH TO APPLY"
        apply_decision = "STRONG MATCH — RECOMMENDED TO APPLY"
        apply_badge_type = "success"
        is_good_match = True
        apply_summary = (
            f"Your resume is a STRONG MATCH at the {percentile}th percentile (Z = {z_score:+.2f}, composite {composite_score}%), "
            f"placing you in the top quartile of the pool (μ={benchmark_mean}, σ={benchmark_std}). Download your Prescriptive SMART PDF Resume and prep for interviews below."
        )
    elif percentile >= 50.0:
        match_strength_label = "MODERATE MATCH — TAILOR & APPLY"
        apply_decision = "MODERATE MATCH — TAILOR RESUME THEN APPLY"
        apply_badge_type = "warning"
        is_good_match = True
        apply_summary = (
            f"Your resume is a MODERATE MATCH at the {percentile}th percentile (Z = {z_score:+.2f}, composite {composite_score}%), "
            f"above the pool average (μ={benchmark_mean}). Use the Prescriptive SMART PDF Resume below to close keyword gaps before applying."
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

    company_brief = build_company_brief(
        jd_text=jd_text,
        company_name=company_name,
        role_title=role_title
    )

    tell_me_pitch = build_tell_me_about_yourself(
        role_title=role_title,
        company_label=company_brief["company_label"],
        unique_matched=unique_matched,
        top_axis=top_axis_row["Category"],
        percentile=percentile,
        exp_info=exp_info
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

    prescriptive_resume = build_prescriptive_smart_resume(
        resume_text=resume_text,
        role_title=role_title,
        company_label=company_brief["company_label"],
        unique_matched=unique_matched,
        axis_details=axis_details,
        top_axis=top_axis_row["Category"],
        use_smart_goals=use_smart_goals
    )

    analytics_stages = {
        "diagnostic": (
            f"Skills Match: {skill_match_score:.1f}% ({len(unique_matched)} matched, {len(unique_missing)} gaps) | "
            f"Contextual Phrasing: {contextual_score}% | Top Axis: {top_axis_row['Category']} ({top_axis_row['Category Match (%)']}%)."
        ),
        "predictive": (
            f"Pool Standing: {percentile}th Percentile (Z = {z_score:+.2f}) | Tier: {tier.split('(')[0].strip()} | "
            f"Projected Ramp-Up: {ramp_up.split('(')[0].strip()} | Retention Outlook: {retention_prob}."
        ),
        "prescriptive": (
            f"Verdict: {match_strength_label} -> Generated a role-tailored {'SMART-Goal ' if use_smart_goals else ''}PDF Resume "
            f"prioritizing {', '.join([s.title() for s in unique_matched[:4]]) or 'core analytics skills'} for {company_brief['company_label']}."
        )
    }

    insights = [
        f"[Diagnostic] Keyword & Competency Balance: Strongest axis is {top_axis_row['Category']} ({top_axis_row['Category Match (%)']}%), while {low_axis_row['Category']} ({low_axis_row['Category Match (%)']}%) is the primary gap area.",
        f"[Diagnostic] HR & People Analytics Signal: People Analytics Index is {people_analytics_index}% (HR/People: {hr_score}%, Workforce/Labor Ops: {ops_score}%, Leadership: {lead_score}%).",
        f"[Predictive] Pool Percentile & Readiness: {composite_score}% composite score places you at the {percentile}th percentile (Z = {z_score:+.2f}), projecting {ramp_up.lower()} and {attrition_risk.lower()} attrition risk.",
        f"[Prescriptive] Resume Optimization Opportunity: Matched {len(unique_matched)} JD skill areas with {contextual_score}% narrative alignment against {len(unique_missing)} unmatched JD skill area(s)."
    ]

    if exp_info["jd_years"] > 0 and exp_info["resume_years"] > 0:
        if exp_info["exp_gap"] < 0:
            insights.append(
                f"[Diagnostic] Tenure Delta: Resume indicates ~{exp_info['resume_years']} years vs. {exp_info['jd_years']}+ years in the JD ({exp_info['exp_gap']} yrs delta)."
            )
        else:
            insights.append(
                f"[Diagnostic] Tenure Alignment: Resume (~{exp_info['resume_years']} yrs) meets or exceeds the {exp_info['jd_years']}+ yrs benchmark ({exp_info['exp_gap']:+d} yrs delta)."
            )

    recommendations = []
    if percentile >= 75.0:
        recommendations.append(f"[Prescriptive Action] Strong {percentile}th-percentile match-download your customized SMART PDF Resume below and submit your application promptly.")
    elif percentile >= 50.0:
        recommendations.append(f"[Prescriptive Action] Moderate {percentile}th-percentile match-use the generated SMART PDF Resume to mirror exact JD phrasing before applying.")
    else:
        recommendations.append(f"[Prescriptive Action] Below-median {percentile}th-percentile match-skip low-ROI applications here and target roles via the Indeed Easy Apply links below.")

    if unique_matched:
        top_m = ", ".join([s.title() for s in unique_matched[:5]])
        recommendations.append(f"Anchor your SMART bullets on quantified outcomes (15% operating cost reduction, automated reporting hours saved) paired with: {top_m}.")

    if unique_missing:
        top_x = ", ".join([s.title() for s in unique_missing[:5]])
        recommendations.append(f"Close visible skill gaps by surfacing adjacent projects or coursework for: {top_x}.")
    else:
        recommendations.append("All detected job description skills are matched-focus interview stories on measurable business ROI.")

    return {
        "match_strength_label": match_strength_label,
        "apply_decision": apply_decision,
        "apply_badge_type": apply_badge_type,
        "apply_summary": apply_summary,
        "is_good_match": is_good_match,
        "analytics_stages": analytics_stages,
        "prescriptive_resume": prescriptive_resume,
        "company_brief": company_brief,
        "tell_me_pitch": tell_me_pitch,
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
