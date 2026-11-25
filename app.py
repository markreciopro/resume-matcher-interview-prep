import re
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from analyzer import (
    extract_text_from_file,
    analyze_candidate_fit,
    build_indeed_easy_apply_links
)

st.set_page_config(
    page_title="Personal Job Match & Interview Prep Studio",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Personal Job Match & Prescriptive Resume Studio")
st.markdown(
    "Powered by **Diagnostic**, **Predictive**, and **Prescriptive Analytics** — evaluate percentile match strength, inspect company and role alignment, and download a customized **SMART-Goal PDF Resume**."
)

with st.sidebar:
    st.header("⚙️ Analytics & Resume Settings")
    st.caption("Configure the Normal Distribution benchmark and Prescriptive PDF Resume options.")
    pool_mean = st.slider("Pool Mean Score (μ)", 40.0, 80.0, 65.0, 1.0)
    pool_std = st.slider("Standard Deviation (σ)", 5.0, 20.0, 12.0, 0.5)
    target_bar = st.slider("Radar Target Threshold (%)", 40.0, 100.0, 75.0, 5.0)
    st.divider()
    use_smart_goals = st.checkbox("Format Tailored Resume with SMART Goals", value=True)
    st.caption("Specific, Measurable, Achievable, Relevant, and Time-Bound bullets.")
    st.divider()
    st.markdown("**Percentile & Match Guide**")
    st.caption("• ≥ 75th pct: Strong Match to Apply\n• 50th–74th pct: Moderate Match (Tailor First)\n• < 50th pct: Not a Strong Match Yet")

st.subheader("🔍 Target Role, Company & Indeed Search Filters")
r_col1, r_col2, r_col3, r_col4 = st.columns([2, 2, 2, 1])
with r_col1:
    target_role = st.text_input("Target Job Title / Role", value="Data and Business Analyst")
with r_col2:
    target_company = st.text_input("Company Name (Optional)", value="", placeholder="Auto-detected from JD if blank")
with r_col3:
    target_location = st.text_input("City, State or Zip (Optional)", value="Las Vegas, NV")
with r_col4:
    st.write("")
    st.write("")
    remote_filter = st.checkbox("Remote / Hybrid", value=False)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. My Resume")
    uploaded_resume = st.file_uploader("Upload Resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
    resume_text_input = st.text_area("Or paste resume text:", height=180, placeholder="Paste your resume text here...")

with col2:
    st.subheader("2. Target Job Description")
    jd_text_input = st.text_area("Paste Job Description:", height=255, placeholder="Paste job description, requirements, and skills...")

analyze_btn = st.button("🚀 Run Diagnostic, Predictive & Prescriptive Analysis", type="primary", use_container_width=True)

if analyze_btn:
    resume_text = extract_text_from_file(uploaded_resume) if uploaded_resume else resume_text_input.strip()
    jd_text = jd_text_input.strip()

    if not resume_text or not jd_text:
        st.error("Please provide both your Resume (upload or paste) and a Job Description.")
    else:
        res = analyze_candidate_fit(
            resume_text=resume_text,
            jd_text=jd_text,
            benchmark_mean=pool_mean,
            benchmark_std=pool_std,
            role_title=target_role,
            company_name=target_company,
            use_smart_goals=use_smart_goals
        )
        indeed_links = build_indeed_easy_apply_links(
            role_query=target_role,
            location=target_location,
            remote_only=remote_filter,
            matched_skills=res["matched_skills"],
            missing_skills=res["missing_skills"]
        )

        st.divider()
        st.subheader("🚦 Percentile-Based Application Verdict")
        verdict_msg = f"**{res['apply_decision']} ({res['percentile']}th Percentile)** — {res['apply_summary']}"
        if res["apply_badge_type"] == "success":
            st.success(verdict_msg)
        elif res["apply_badge_type"] == "warning":
            st.warning(verdict_msg)
        else:
            st.error(verdict_msg)

        # 3-Stage Data Analytics Framework Banner
        st.subheader("🧭 Data Analytics Pipeline: Diagnostic → Predictive → Prescriptive")
        da1, da2, da3 = st.columns(3)
        with da1:
            st.info(f"**1. Diagnostic Analytics (What & Why)**\n\n{res['analytics_stages']['diagnostic']}")
        with da2:
            st.warning(f"**2. Predictive Analytics (Pool Outlook)**\n\n{res['analytics_stages']['predictive']}")
        with da3:
            st.success(f"**3. Prescriptive Analytics (Action Plan)**\n\n{res['analytics_stages']['prescriptive']}")

        st.divider()

        # Prescriptive Analytics Deliverable: Tailored SMART-Goal PDF Resume Download
        st.subheader("📝 Prescriptive Analytics: Tailored PDF Resume Generator")
        if res["is_good_match"]:
            st.caption(
                f"Because this role is a {res['match_strength_label']} ({res['percentile']}th percentile), "
                f"your resume has been customized for **{target_role}** at **{res['company_brief']['company_label']}** "
                + ("using **SMART Goal** impact bullets." if use_smart_goals else "with role-prioritized skills.")
            )
        else:
            st.caption(
                "This posting is currently below the 50th percentile, but you can still preview or download a tailored draft below."
            )

        safe_role_slug = re.sub(r'[^A-Za-z0-9]+', '_', target_role).strip('_') or "Target_Role"
        pdf_filename = f"Mark_Recio_Resume_{safe_role_slug}.pdf"

        st.download_button(
            label=f"⬇️ Download Customized {'SMART-Goal ' if use_smart_goals else ''}PDF Resume ({pdf_filename})",
            data=res["prescriptive_resume"]["pdf_bytes"],
            file_name=pdf_filename,
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

        with st.expander("👀 Preview Customized Resume Content Before Downloading", expanded=False):
            st.markdown(res["prescriptive_resume"]["preview_md"])

        st.divider()

        # About the Company + Tell Me About Yourself (Matched to Role)
        comp_col, pitch_col = st.columns(2)

        with comp_col:
            st.subheader(f"🏢 About the Company: {res['company_brief']['company_label']}")
            st.info(res["company_brief"]["theme_summary"])
            if res["company_brief"]["jd_excerpt"]:
                st.caption("Excerpt detected from posting:")
                st.write(f"> *{res['company_brief']['jd_excerpt']}*")
            st.markdown("**Pre-Interview Company Research Checklist:**")
            for chk in res["company_brief"]["research_checks"]:
                st.markdown(f"- {chk}")

        with pitch_col:
            st.subheader("🗣️ 'Tell Me About Yourself' — Matched to This Role")
            st.success(res["tell_me_pitch"]["pitch_script"])
            st.markdown("**How to Deliver It:**")
            for pil in res["tell_me_pitch"]["key_pillars"]:
                st.markdown(f"- {pil}")

        st.divider()

        # Side-by-Side Highlighted Resume vs. Pasted Job Description
        st.subheader("📄 Diagnostic View: Side-by-Side Resume & Job Description Match")
        st.markdown(
            "Highlight Legend: "
            "<mark style='background-color: rgba(0, 204, 150, 0.28); border-bottom: 2px solid #00CC96; padding: 1px 6px; border-radius: 4px; color: inherit;'><b>Green = Matched in Both</b></mark> &nbsp;|&nbsp; "
            "<mark style='background-color: rgba(239, 85, 59, 0.28); border-bottom: 2px solid #EF553B; padding: 1px 6px; border-radius: 4px; color: inherit;'><b>Red = Missing from Resume (In JD)</b></mark> &nbsp;|&nbsp; "
            "<mark style='background-color: rgba(99, 110, 250, 0.25); border-bottom: 2px solid #636EFA; padding: 1px 6px; border-radius: 4px; color: inherit;'><b>Blue = Additional Resume Skills</b></mark>",
            unsafe_allow_html=True
        )

        doc_col1, doc_col2 = st.columns(2)
        with doc_col1:
            st.markdown("#### 👤 Your Parsed Resume (Highlighted)")
            st.markdown(
                f"<div style='height: 380px; overflow-y: auto; padding: 14px; border: 1px solid rgba(128,128,128,0.35); border-radius: 8px; line-height: 1.6; font-size: 14px;'>"
                f"{res['resume_highlighted_html']}"
                f"</div>",
                unsafe_allow_html=True
            )
        with doc_col2:
            st.markdown("#### 📋 Pasted Job Description (Highlighted)")
            st.markdown(
                f"<div style='height: 380px; overflow-y: auto; padding: 14px; border: 1px solid rgba(128,128,128,0.35); border-radius: 8px; line-height: 1.6; font-size: 14px;'>"
                f"{res['jd_highlighted_html']}"
                f"</div>",
                unsafe_allow_html=True
            )

        st.divider()

        # Row 1: Predictive Bell Curve + Diagnostic 6-Axis Radar
        c_bell, c_radar = st.columns([3, 2])

        with c_bell:
            st.subheader("🔔 Predictive Analytics: Normal Distribution Bell Curve")
            x = np.linspace(max(0, pool_mean - 4 * pool_std), min(100, pool_mean + 4 * pool_std), 600)
            y = norm.pdf(x, pool_mean, pool_std)

            fig_bell = go.Figure()

            fig_bell.add_trace(go.Scatter(
                x=x, y=y, mode="lines", name="Talent Pool Bell Curve",
                line=dict(color="#636EFA", width=2.5),
                fill="tozeroy", fillcolor="rgba(99, 110, 250, 0.10)"
            ))

            x_cdf = x[x <= res["composite_score"]]
            y_cdf = norm.pdf(x_cdf, pool_mean, pool_std)
            fig_bell.add_trace(go.Scatter(
                x=x_cdf, y=y_cdf, mode="lines", name=f"Percentile Area ({res['percentile']}th pct)",
                line=dict(width=0),
                fill="tozeroy", fillcolor="rgba(0, 204, 150, 0.25)"
            ))

            for mult, label, dash in [
                (-1, "-1σ", "dot"),
                (0, f"μ ({pool_mean})", "dash"),
                (1, "+1σ", "dot"),
                (2, "+2σ", "dot")
            ]:
                val = pool_mean + mult * pool_std
                if 0 <= val <= 100:
                    fig_bell.add_vline(
                        x=val, line_dash=dash, line_color="#888888", line_width=1.2,
                        annotation_text=label, annotation_position="top"
                    )

            fig_bell.add_vline(
                x=res["composite_score"], line_dash="solid", line_color="#EF553B", line_width=3,
                annotation_text=f"You ({res['percentile']}th pct | {res['composite_score']}%)",
                annotation_position="top right"
            )

            fig_bell.update_layout(
                xaxis_title="Fit & Readiness Score (0–100)",
                yaxis_title="Probability Density",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=45, b=20),
                height=420
            )
            st.plotly_chart(fig_bell, use_container_width=True)

        with c_radar:
            st.subheader("🕸️ Diagnostic 6-Axis Competency Radar")
            cat_df = res["category_df"]
            vals = cat_df["Category Match (%)"].tolist()
            cats = cat_df["Category"].tolist()
            target_vals = [target_bar] * len(cats)

            hover_texts = []
            for row in cat_df.itertuples():
                hover_texts.append(
                    f"<b>{row.Category}</b><br>"
                    f"Match Score: {getattr(row, '_6')}%<br>"
                    f"Matched / JD Required: {row.Matched} / {getattr(row, '_2')}<br>"
                    f"Resume Breadth: {getattr(row, '_5')} skills"
                )

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=target_vals + [target_vals[0]],
                theta=cats + [cats[0]],
                mode="lines",
                line=dict(color="#FFA15A", width=2, dash="dash"),
                name=f"Target Benchmark ({target_bar:.0f}%)"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                text=hover_texts + [hover_texts[0]],
                hoverinfo="text",
                fill="toself",
                fillcolor="rgba(0, 204, 150, 0.25)",
                line=dict(color="#00CC96", width=2.5),
                name="Your Score (%)"
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
                margin=dict(l=45, r=45, t=55, b=35),
                height=420
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.divider()

        # Row 2: Improved Heatmap + Unclipped Donut Chart
        c_heat, c_donut = st.columns([3, 2])

        with c_heat:
            st.subheader("🔥 Diagnostic Competency & Skill Coverage Heatmap")
            short_cats = [
                "Data & BI",
                "Analytics & Forecast",
                "HR & People",
                "Workforce & Ops",
                "Process & Comp.",
                "Leadership"
            ]

            max_matched = max(1, int(cat_df["Matched"].max()))
            max_breadth = max(1, int(cat_df["Resume Breadth"].max()))
            max_missing = max(1, int(cat_df["Missing"].max()))

            heat_z = [
                cat_df["Category Match (%)"].tolist(),
                [(v / max_matched) * 100 for v in cat_df["Matched"]],
                [(v / max_breadth) * 100 for v in cat_df["Resume Breadth"]],
                [(v / max_missing) * 100 for v in cat_df["Missing"]]
            ]
            heat_text = [
                [f"{v:.0f}%" for v in cat_df["Category Match (%)"]],
                [f"{v}" for v in cat_df["Matched"]],
                [f"{v}" for v in cat_df["Resume Breadth"]],
                [f"{v}" for v in cat_df["Missing"]]
            ]
            y_labels = [
                "Axis Match (%)",
                "Matched JD (#)",
                "Resume Breadth (#)",
                "Missing Gaps (#)"
            ]

            fig_heat = go.Figure(data=go.Heatmap(
                z=heat_z,
                x=short_cats,
                y=y_labels,
                text=heat_text,
                texttemplate="<b>%{text}</b>",
                textfont={"size": 13, "color": "#111111"},
                xgap=4,
                ygap=4,
                colorscale=[
                    [0.0, "#E8F5E9"],
                    [0.5, "#80CBC4"],
                    [1.0, "#00897B"]
                ],
                zmin=0,
                zmax=100,
                showscale=True,
                colorbar=dict(title="Relative<br>Level", ticksuffix="%")
            ))
            fig_heat.update_layout(
                height=370,
                margin=dict(l=15, r=15, t=20, b=30),
                xaxis=dict(side="bottom", tickangle=0),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        with c_donut:
            st.subheader("🍩 Skill Alignment Breakdown")
            matched_count = len(res["matched_skills"])
            missing_count = len(res["missing_skills"])
            total_resume_skills = sum(len(v["resume_has"]) for v in res["axis_details"].values())
            bonus_resume_skills = max(0, total_resume_skills - matched_count)

            donut_labels = [
                "Matched to JD",
                "Additional Resume Skills",
                "Missing JD Gaps"
            ]
            donut_values = [
                matched_count,
                bonus_resume_skills,
                missing_count
            ]

            if sum(donut_values) == 0:
                donut_values = [1, 0, 0]

            fig_donut = go.Figure(data=[go.Pie(
                labels=donut_labels,
                values=donut_values,
                hole=0.60,
                sort=False,
                marker=dict(
                    colors=["#00CC96", "#636EFA", "#EF553B"],
                    line=dict(color="#181818", width=2)
                ),
                textinfo="value+percent",
                textposition="inside",
                insidetextorientation="horizontal",
                textfont=dict(size=12, color="white")
            )])
            fig_donut.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.05,
                    xanchor="center",
                    x=0.5
                ),
                height=370,
                margin=dict(l=15, r=15, t=20, b=50),
                annotations=[dict(
                    text=f"<b>{res['skill_match_score']:.0f}%</b><br><span style='font-size:11px'>JD Fit</span>",
                    x=0.5, y=0.5, font_size=16, showarrow=False
                )]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # Row 3: Domain Counts Bar Chart + Summary Table
        st.subheader("📊 Domain Counts Comparison")
        c_bar, c_tbl = st.columns([3, 2])

        with c_bar:
            fig_bar = px.bar(
                cat_df,
                x="Category",
                y=["Matched", "JD Required", "Missing"],
                barmode="group",
                color_discrete_sequence=["#00CC96", "#636EFA", "#EF553B"],
                text_auto=True
            )
            fig_bar.update_layout(height=300, margin=dict(l=15, r=15, t=20, b=20), xaxis_tickangle=-15)
            st.plotly_chart(fig_bar, use_container_width=True)

        with c_tbl:
            st.dataframe(cat_df, use_container_width=True, hide_index=True)

        # Interview Readiness & Focus Points
        st.divider()
        st.subheader("🎤 Interview Invite Preparation: Focus Points & Questions to Ask")
        if res["is_good_match"]:
            st.caption(f"At the {res['percentile']}th percentile ({res['match_strength_label']}), you are well-positioned for an interview invite. Use these tailored talking points and questions:")
            ip_col1, ip_col2 = st.columns(2)
            with ip_col1:
                st.markdown("#### 📌 Key Focus Points & Talking Stories")
                for fp in res["interview_prep"]["focus_points"]:
                    st.info(fp)
            with ip_col2:
                st.markdown("#### ❓ Smart Questions to Ask the Interviewer")
                for q in res["interview_prep"]["questions_to_ask"]:
                    st.success(q)
        else:
            st.warning(
                f"Currently at the {res['percentile']}th percentile ({res['match_strength_label']}). Tailor your resume to the missing skills below or explore higher-fit roles via Indeed Easy Apply."
            )

        # Insights & Recommendations
        st.divider()
        ins_col, rec_col = st.columns(2)
        with ins_col:
            st.subheader("🔍 Diagnostic, Predictive & Prescriptive Insights")
            for item in res["insights"]:
                st.info(item)

        with rec_col:
            st.subheader("💡 Recommended Next Steps")
            for rec in res["recommendations"]:
                st.success(rec)

        # Matched vs Missing Skills Overall
        st.divider()
        st.subheader("📋 Matched & Missing Skills Inventory")
        m_col, x_col = st.columns(2)
        with m_col:
            st.markdown("✅ **Matched Skills**")
            if res["matched_skills"]:
                st.write(", ".join([f"`{s}`" for s in res["matched_skills"]]))
            else:
                st.write("No taxonomy skills matched yet.")
        with x_col:
            st.markdown("⚠️ **Missing Skills (Present in JD)**")
            if res["missing_skills"]:
                st.write(", ".join([f"`{s}`" for s in res["missing_skills"]]))
            else:
                st.write("All detected job description skills are matched.")

        # Indeed Easy Apply Links Section
        st.divider()
        st.subheader("🔗 Indeed 'Easily Apply' Matching Job Links")
        st.caption("Each link opens Indeed pre-filtered to Easy Apply (`sritch=1`) using your target role, location, and skill profile.")

        for row in indeed_links:
            st.markdown(f"- **[{row['Link Title']}]({row['Indeed Easy Apply URL']})** — *{row['Why This Link']}* (`{row['Search Terms']}` | `{row['Location Filter']}`)")

        st.dataframe(
            pd.DataFrame(indeed_links),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Indeed Easy Apply URL": st.column_config.LinkColumn("Indeed Easy Apply URL", display_text="Open on Indeed ↗")
            }
        )

        # Export Summary Report
        st.divider()
        st.subheader("📥 Export Full Analytics & Interview Prep Guide")
        prep_md = ""
        if res["is_good_match"]:
            prep_md = (
                "## Interview Focus Points\n"
                + "\n".join([f"- {fp}" for fp in res["interview_prep"]["focus_points"]])
                + "\n\n## Questions to Ask the Interviewer\n"
                + "\n".join([f"- {q}" for q in res["interview_prep"]["questions_to_ask"]])
                + "\n\n"
            )

        report_md = (
            f"# Personal Job Match & Prescriptive Resume Report\n\n"
            f"- **Percentile Match Verdict:** {res['apply_decision']} ({res['percentile']}th Percentile)\n"
            f"- **Verdict Rationale:** {res['apply_summary']}\n"
            f"- **Company:** {res['company_brief']['company_label']}\n"
            f"- **Target Role:** {target_role} ({target_location or 'Any Location'})\n"
            f"- **Composite Fit:** {res['composite_score']}% (Z-Score: {res['z_score']:+.2f}, {res['percentile']}th Percentile)\n"
            f"- **Skills Match:** {res['skill_match_score']}% | **Context Match:** {res['contextual_score']}%\n"
            f"- **HR & People Index:** {res['people_analytics_index']}%\n"
            f"- **Projected Tier:** {res['tier']}\n\n"
            f"## 3-Stage Data Analytics Summary\n"
            f"- **Diagnostic:** {res['analytics_stages']['diagnostic']}\n"
            f"- **Predictive:** {res['analytics_stages']['predictive']}\n"
            f"- **Prescriptive:** {res['analytics_stages']['prescriptive']}\n\n"
            f"## About the Company\n{res['company_brief']['theme_summary']}\n\n"
            f"## Tell Me About Yourself (Matched to Role)\n{res['tell_me_pitch']['pitch_script']}\n\n"
            f"## Matched Skills\n{', '.join(res['matched_skills']) or 'None'}\n\n"
            f"## Missing JD Skills\n{', '.join(res['missing_skills']) or 'None'}\n\n"
            f"{prep_md}"
            f"## Detailed Insights\n" + "\n".join([f"- {i}" for i in res["insights"]]) + "\n\n"
            f"## Recommended Next Steps\n" + "\n".join([f"- {r}" for r in res["recommendations"]]) + "\n"
        )
        ex1, ex2 = st.columns(2)
        ex1.download_button(
            "📄 Download Interview & Analytics Guide (.md)",
            data=report_md,
            file_name="job_match_interview_prep.md",
            mime="text/markdown",
            use_container_width=True
        )
        ex2.download_button(
            "📊 Download Domain Breakdown (.csv)",
            data=cat_df.to_csv(index=False),
            file_name="competency_breakdown.csv",
            mime="text/csv",
            use_container_width=True
        )
