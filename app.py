"""
Creative Scorer Lite — Image Compliance & AI Performance Scoring
Built by Manus for Athena · Media Strategy

HOW TO USE:
1. Enter your OpenAI API key in the sidebar (required for AI scoring).
2. Select your target platforms, audience stage, and market.
3. Upload 1–5 image assets (JPG, PNG, or WebP).
4. Click "Score Creatives" to run compliance checks and AI scoring.
5. Review per-asset scorecards and the batch summary table.
6. Switch to the "Compliance Legend" tab to understand every check.
"""

import io
import base64
import math
import re
import streamlit as st
from PIL import Image
import numpy as np
import openai

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Creative Scorer Lite",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ────────────────────────────────────────────────────────────────

PLATFORMS = ["Meta", "TikTok", "Google PMax", "YouTube"]
AUDIENCE_STAGES = ["Awareness", "Consideration", "Conversion", "Retention"]
MAX_ASSETS = 5

# Platform specs: (max_size_mb, min_w, min_h, preferred_ratios, acceptable_ratios, max_text_pct)
PLATFORM_SPECS = {
    "Meta": {
        "max_size_mb": 30,
        "min_w": 1080, "min_h": 1080,
        "preferred_ratios": [(1, 1), (4, 5), (9, 16)],
        "acceptable_ratios": [(16, 9)],
        "max_text_pct": 20,
        "formats": ["jpg", "jpeg", "png", "webp"],
        "safe_top_pct": 14, "safe_bottom_pct": 20,
    },
    "TikTok": {
        "max_size_mb": 0.5,
        "min_w": 720, "min_h": 1280,
        "preferred_ratios": [(9, 16)],
        "acceptable_ratios": [(1, 1)],
        "max_text_pct": 25,
        "formats": ["jpg", "jpeg", "png", "webp"],
        "safe_top_pct": 15, "safe_bottom_pct": 25,
    },
    "Google PMax": {
        "max_size_mb": 5,
        "min_w": 600, "min_h": 314,
        "preferred_ratios": [(191, 100)],
        "acceptable_ratios": [(1, 1), (4, 3)],
        "max_text_pct": 20,
        "formats": ["jpg", "jpeg", "png", "webp"],
        "safe_top_pct": 10, "safe_bottom_pct": 10,
    },
    "YouTube": {
        "max_size_mb": 2,
        "min_w": 1280, "min_h": 720,
        "preferred_ratios": [(16, 9)],
        "acceptable_ratios": [(1, 1)],
        "max_text_pct": 20,
        "formats": ["jpg", "jpeg", "png", "webp"],
        "safe_top_pct": 10, "safe_bottom_pct": 20,
    },
}

DIMENSION_WEIGHTS = {
    "Hook Strength": 0.25,
    "Message Clarity": 0.25,
    "Mobile Fit": 0.20,
    "Audience Fit": 0.15,
    "Native Feel": 0.15,
}

# ─── Compliance engine ────────────────────────────────────────────────────────

def ratio_matches(w: int, h: int, ratios: list, tolerance: float = 0.05) -> bool:
    actual = w / h if h > 0 else 0
    for rw, rh in ratios:
        target = rw / rh
        if abs(actual - target) / target <= tolerance:
            return True
    return False


def estimate_text_density(img: Image.Image) -> float:
    """Rough text density estimate via edge detection on grayscale."""
    gray = np.array(img.convert("L").resize((200, 200)))
    # Sobel-like edge detection
    gx = np.abs(np.diff(gray.astype(np.int16), axis=1))
    gy = np.abs(np.diff(gray.astype(np.int16), axis=0))
    edges = (gx[:199, :] + gy[:, :199]) / 2
    edge_ratio = float(np.sum(edges > 30)) / edges.size
    # Heuristic: high-frequency edges correlate with text
    return min(edge_ratio * 180, 100)


def run_compliance(platform: str, img: Image.Image, file_size_mb: float, fmt: str) -> list[dict]:
    spec = PLATFORM_SPECS[platform]
    checks = []
    w, h = img.size

    # 1. File format
    if fmt.lower() in spec["formats"]:
        checks.append({"check": "File Format", "status": "PASS", "detail": f"{fmt.upper()} accepted"})
    else:
        checks.append({"check": "File Format", "status": "FAIL", "detail": f"{fmt.upper()} not accepted by {platform}"})

    # 2. File size
    if file_size_mb <= spec["max_size_mb"]:
        checks.append({"check": "File Size", "status": "PASS", "detail": f"{file_size_mb:.2f} MB ≤ {spec['max_size_mb']} MB limit"})
    else:
        checks.append({"check": "File Size", "status": "FAIL", "detail": f"{file_size_mb:.2f} MB exceeds {spec['max_size_mb']} MB limit"})

    # 3. Resolution
    if w >= spec["min_w"] and h >= spec["min_h"]:
        checks.append({"check": "Resolution", "status": "PASS", "detail": f"{w}×{h} px meets minimum {spec['min_w']}×{spec['min_h']} px"})
    else:
        checks.append({"check": "Resolution", "status": "FAIL", "detail": f"{w}×{h} px below minimum {spec['min_w']}×{spec['min_h']} px"})

    # 4. Aspect ratio
    if ratio_matches(w, h, spec["preferred_ratios"]):
        checks.append({"check": "Aspect Ratio", "status": "PASS", "detail": f"{w}:{h} matches a preferred ratio"})
    elif ratio_matches(w, h, spec["acceptable_ratios"]):
        checks.append({"check": "Aspect Ratio", "status": "RISK", "detail": f"{w}:{h} is acceptable but not preferred — may reduce delivery efficiency"})
    else:
        checks.append({"check": "Aspect Ratio", "status": "FAIL", "detail": f"{w}:{h} does not match any accepted ratio for {platform}"})

    # 5. Text density
    text_pct = estimate_text_density(img)
    if text_pct < spec["max_text_pct"] * 0.8:
        checks.append({"check": "Text Density", "status": "PASS", "detail": f"Estimated ~{text_pct:.0f}% text coverage (limit {spec['max_text_pct']}%)"})
    elif text_pct < spec["max_text_pct"]:
        checks.append({"check": "Text Density", "status": "RISK", "detail": f"Estimated ~{text_pct:.0f}% — near the {spec['max_text_pct']}% limit, manual review recommended"})
    else:
        checks.append({"check": "Text Density", "status": "FAIL", "detail": f"Estimated ~{text_pct:.0f}% exceeds {spec['max_text_pct']}% limit"})

    # 6. Safe zone (always manual review)
    checks.append({
        "check": "Safe Zone",
        "status": "RISK",
        "detail": f"Manual review required — keep key elements away from top {spec['safe_top_pct']}% and bottom {spec['safe_bottom_pct']}% of frame",
    })

    return checks


def overall_status(checks: list[dict]) -> str:
    statuses = [c["status"] for c in checks]
    if "FAIL" in statuses:
        return "FAIL"
    if "RISK" in statuses:
        return "RISK"
    return "PASS"


# ─── AI scoring ───────────────────────────────────────────────────────────────

def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def score_with_ai(img: Image.Image, platforms: list[str], audience_stage: str, market: str, api_key: str) -> dict:
    client = openai.OpenAI(api_key=api_key)
    platform_str = ", ".join(platforms)
    prompt = f"""You are an expert digital advertising creative strategist. Evaluate this ad creative image for {platform_str} targeting {audience_stage} audiences in {market}.

Score each dimension from 1–10 (integers only). Be critical and realistic — reserve 9–10 for exceptional work.

Respond ONLY with valid JSON in exactly this structure:
{{
  "hook_strength": {{"score": <1-10>, "rationale": "<1 sentence>", "recommendation": "<1 actionable fix>"}},
  "message_clarity": {{"score": <1-10>, "rationale": "<1 sentence>", "recommendation": "<1 actionable fix>"}},
  "mobile_fit": {{"score": <1-10>, "rationale": "<1 sentence>", "recommendation": "<1 actionable fix>"}},
  "audience_fit": {{"score": <1-10>, "rationale": "<1 sentence>", "recommendation": "<1 actionable fix>"}},
  "native_feel": {{"score": <1-10>, "rationale": "<1 sentence>", "recommendation": "<1 actionable fix>"}},
  "policy_compliance": {{"status": "PASS" | "RISK" | "FAIL", "note": "<1 sentence>"}},
  "brand_compliance": {{"status": "PASS" | "RISK" | "FAIL", "note": "<1 sentence>"}}
}}"""

    data_url = image_to_data_url(img)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                {"type": "text", "text": prompt},
            ]}
        ],
        max_tokens=800,
        temperature=0.3,
    )

    raw = response.choices[0].message.content or ""
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    import json
    return json.loads(raw)


def compute_performance_score(ai_result: dict) -> float:
    total = 0.0
    key_map = {
        "Hook Strength": "hook_strength",
        "Message Clarity": "message_clarity",
        "Mobile Fit": "mobile_fit",
        "Audience Fit": "audience_fit",
        "Native Feel": "native_feel",
    }
    for dim, weight in DIMENSION_WEIGHTS.items():
        key = key_map[dim]
        score = ai_result.get(key, {}).get("score", 5)
        total += (score / 10) * 100 * weight
    return round(total, 1)


def verdict(score: float) -> tuple[str, str]:
    if score >= 80:
        return "Strong", "🟢"
    if score >= 60:
        return "Run with Revisions", "🟡"
    return "Rework", "🔴"


# ─── UI helpers ───────────────────────────────────────────────────────────────

STATUS_EMOJI = {"PASS": "✅", "RISK": "⚠️", "FAIL": "❌"}
STATUS_COLOR = {"PASS": "green", "RISK": "orange", "FAIL": "red"}

def status_badge(status: str) -> str:
    return f":{STATUS_COLOR[status]}[**{STATUS_EMOJI[status]} {status}**]"


def render_scorecard(name: str, img: Image.Image, checks_by_platform: dict, ai_result: dict | None, perf_score: float | None, selected_platforms: list[str]):
    v_label, v_emoji = verdict(perf_score) if perf_score is not None else ("—", "")

    with st.container(border=True):
        col_img, col_meta = st.columns([1, 3])

        with col_img:
            st.image(img, use_container_width=True)

        with col_meta:
            st.markdown(f"### {name}")
            if perf_score is not None:
                score_col, verdict_col, policy_col, brand_col = st.columns(4)
                with score_col:
                    st.metric("Performance Score", f"{perf_score}/100")
                with verdict_col:
                    st.markdown(f"**Verdict**  \n{v_emoji} {v_label}")
                if ai_result:
                    with policy_col:
                        pc = ai_result.get("policy_compliance", {})
                        st.markdown(f"**Policy**  \n{STATUS_EMOJI.get(pc.get('status','—'), '—')} {pc.get('status','—')}")
                    with brand_col:
                        bc = ai_result.get("brand_compliance", {})
                        st.markdown(f"**Brand**  \n{STATUS_EMOJI.get(bc.get('status','—'), '—')} {bc.get('status','—')}")

        # Tabs
        tab_labels = ["Compliance"] + (["AI Dimensions", "Recommendations"] if ai_result else [])
        tabs = st.tabs(tab_labels)

        # Compliance tab
        with tabs[0]:
            for platform in selected_platforms:
                checks = checks_by_platform.get(platform, [])
                ov = overall_status(checks)
                st.markdown(f"**{platform}** — {status_badge(ov)}")
                for c in checks:
                    emoji = STATUS_EMOJI[c["status"]]
                    color = STATUS_COLOR[c["status"]]
                    st.markdown(f"&nbsp;&nbsp;&nbsp;{emoji} **{c['check']}** — :{color}[{c['status']}] — {c['detail']}")
                st.divider()

        # Dimensions tab
        if ai_result and len(tabs) > 1:
            with tabs[1]:
                dim_map = [
                    ("Hook Strength", "hook_strength", "⚡"),
                    ("Message Clarity", "message_clarity", "🎯"),
                    ("Mobile Fit", "mobile_fit", "📱"),
                    ("Audience Fit", "audience_fit", "👥"),
                    ("Native Feel", "native_feel", "✨"),
                ]
                for label, key, icon in dim_map:
                    d = ai_result.get(key, {})
                    score = d.get("score", 0)
                    rationale = d.get("rationale", "")
                    weight = int(DIMENSION_WEIGHTS[label] * 100)
                    col_a, col_b = st.columns([1, 3])
                    with col_a:
                        st.metric(f"{icon} {label} ({weight}%)", f"{score}/10")
                    with col_b:
                        st.markdown(f"*{rationale}*")
                    st.progress(score / 10)

        # Recommendations tab
        if ai_result and len(tabs) > 2:
            with tabs[2]:
                recs = []
                for key in ["hook_strength", "message_clarity", "mobile_fit", "audience_fit", "native_feel"]:
                    rec = ai_result.get(key, {}).get("recommendation", "")
                    if rec:
                        recs.append(rec)
                top3 = recs[:3]
                for i, rec in enumerate(top3, 1):
                    st.markdown(f"**{i}.** {rec}")
                if ai_result.get("policy_compliance", {}).get("note"):
                    st.info(f"**Policy note:** {ai_result['policy_compliance']['note']}")
                if ai_result.get("brand_compliance", {}).get("note"):
                    st.info(f"**Brand note:** {ai_result['brand_compliance']['note']}")


def render_legend():
    st.subheader("Compliance Status Guide")
    cols = st.columns(3)
    with cols[0]:
        st.success("✅ **PASS** — Meets all platform requirements. No action needed.")
    with cols[1]:
        st.warning("⚠️ **RISK** — Acceptable but not optimal, or requires manual verification. Fix before launch.")
    with cols[2]:
        st.error("❌ **FAIL** — Does not meet requirements. Ad will be rejected. Must fix.")

    st.divider()
    st.subheader("Image Compliance Checks")

    checks_data = [
        {
            "Check": "File Format",
            "Meta": "JPG, PNG, GIF, WebP",
            "TikTok": "JPG, PNG, WebP",
            "Google PMax": "JPG, PNG, GIF, WebP",
            "YouTube": "JPG, PNG, GIF, WebP",
            "PASS when": "Format is in the allowed list",
            "RISK when": "—",
            "FAIL when": "Format not accepted by platform",
        },
        {
            "Check": "File Size",
            "Meta": "≤ 30 MB",
            "TikTok": "≤ 500 KB (feed)",
            "Google PMax": "≤ 5 MB",
            "YouTube": "≤ 2 MB (thumbnail)",
            "PASS when": "Within platform limit",
            "RISK when": "—",
            "FAIL when": "Exceeds platform file size cap",
        },
        {
            "Check": "Resolution",
            "Meta": "Min 1080×1080 px",
            "TikTok": "Min 720×1280 px",
            "Google PMax": "Min 600×314 px",
            "YouTube": "Min 1280×720 px",
            "PASS when": "Both dimensions meet minimum",
            "RISK when": "—",
            "FAIL when": "Width or height below minimum",
        },
        {
            "Check": "Aspect Ratio",
            "Meta": "1:1, 4:5, 9:16 preferred; 16:9 acceptable",
            "TikTok": "9:16 preferred; 1:1 acceptable",
            "Google PMax": "1.91:1 preferred; 1:1, 4:3 acceptable",
            "YouTube": "16:9 preferred; 1:1 acceptable",
            "PASS when": "Matches a preferred ratio (±5%)",
            "RISK when": "Matches an acceptable but non-preferred ratio",
            "FAIL when": "Does not match any accepted ratio",
        },
        {
            "Check": "Text Density",
            "Meta": "< 20% text coverage",
            "TikTok": "< 25% text coverage",
            "Google PMax": "< 20% text coverage",
            "YouTube": "< 20% text coverage",
            "PASS when": "Estimated coverage clearly below threshold",
            "RISK when": "Coverage near threshold — manual review recommended",
            "FAIL when": "Coverage clearly exceeds threshold",
        },
        {
            "Check": "Safe Zone",
            "Meta": "Top 14%, bottom 20% reserved",
            "TikTok": "Top 15%, bottom 25%, right 15% reserved",
            "Google PMax": "Top/bottom 10% reserved",
            "YouTube": "Top/bottom 10–20% reserved",
            "PASS when": "—",
            "RISK when": "Always — manual visual verification required",
            "FAIL when": "—",
        },
    ]

    import pandas as pd
    df = pd.DataFrame(checks_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("AI Score Dimensions")

    dim_data = [
        {"Dimension": "⚡ Hook Strength", "Weight": "25%", "What it measures": "Visual impact and attention-grab in the first frame"},
        {"Dimension": "🎯 Message Clarity", "Weight": "25%", "What it measures": "Single primary message is clear; CTA is visible and actionable"},
        {"Dimension": "📱 Mobile Fit", "Weight": "20%", "What it measures": "Text readable at mobile size; faces/objects properly scaled; sufficient contrast"},
        {"Dimension": "👥 Audience Fit", "Weight": "15%", "What it measures": "Creative speaks to the selected audience stage and target market"},
        {"Dimension": "✨ Native Feel", "Weight": "15%", "What it measures": "Looks like organic content rather than an obvious ad; platform-appropriate style"},
    ]
    import pandas as pd
    st.dataframe(pd.DataFrame(dim_data), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Score → Verdict Mapping")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("🟢 **Strong** — Score 80–100")
    with col2:
        st.warning("🟡 **Run with Revisions** — Score 60–79")
    with col3:
        st.error("🔴 **Rework** — Score 0–59")

    st.divider()
    st.info("""
**Notes:**
- Specs reflect 2026 platform policies. Always verify against official platform help centres before launch.
- Safe Zone is always flagged as RISK — it requires manual visual verification of element placement.
- Text Density is estimated via pixel edge analysis — complex layouts may require manual review.
- AI scoring requires an OpenAI API key. Without it, only compliance checks run.
""")


# ─── Main app ─────────────────────────────────────────────────────────────────

def main():
    st.markdown("""
    <style>
    .main-header { font-size: 1.8rem; font-weight: 700; margin-bottom: 0; }
    .sub-header { color: #8B8FA8; font-size: 0.9rem; margin-top: 0; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎯 Creative Scorer Lite")
        st.caption("Image compliance & AI scoring · Built by Manus for Athena")
        st.divider()

        sidebar_tab = st.radio("View", ["⚙️ Settings", "📖 Compliance Legend"], label_visibility="collapsed")

        if sidebar_tab == "⚙️ Settings":
            st.markdown("### Settings")

            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Required for AI performance scoring. Compliance checks run without it.",
            )

            st.markdown("**Target Platform(s)**")
            selected_platforms = []
            cols = st.columns(2)
            for i, p in enumerate(PLATFORMS):
                with cols[i % 2]:
                    if st.checkbox(p, value=(p == "Meta"), key=f"plat_{p}"):
                        selected_platforms.append(p)
            if not selected_platforms:
                selected_platforms = ["Meta"]

            audience_stage = st.selectbox("Audience Stage", AUDIENCE_STAGES)
            market = st.text_input("Target Market", value="Southeast Asia", placeholder="e.g. Southeast Asia")

            st.divider()
            st.markdown("**Quick Reference**")
            st.success("🟢 Strong — 80–100")
            st.warning("🟡 Run with Revisions — 60–79")
            st.error("🔴 Rework — 0–59")

        else:
            # Legend view in sidebar — just a pointer, full legend in main area
            st.info("The full Compliance Legend is shown in the main panel on the right.")
            selected_platforms = ["Meta"]
            audience_stage = "Awareness"
            market = "Southeast Asia"
            api_key = ""

    # ── Main area ─────────────────────────────────────────────────────────────
    if sidebar_tab == "📖 Compliance Legend":
        st.markdown('<p class="main-header">📖 Compliance Legend</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Reference guide for all compliance checks, thresholds, and scoring dimensions</p>', unsafe_allow_html=True)
        render_legend()
        return

    # Settings view — main scoring UI
    st.markdown('<p class="main-header">🎯 Creative Scorer Lite</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload image assets · Run compliance checks · Get AI performance scores</p>', unsafe_allow_html=True)

    st.info("""
**How to use:**
1. Enter your OpenAI API key in the sidebar (optional — compliance checks run without it).
2. Select platforms, audience stage, and market.
3. Upload 1–5 images (JPG, PNG, or WebP).
4. Click **Score Creatives**.
""")

    uploaded_files = st.file_uploader(
        "Upload creatives (JPG, PNG, WebP · up to 5 assets)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Upload 1 to 5 image assets for scoring.",
    )

    if uploaded_files and len(uploaded_files) > MAX_ASSETS:
        st.warning(f"Maximum {MAX_ASSETS} assets per batch. Only the first {MAX_ASSETS} will be scored.")
        uploaded_files = uploaded_files[:MAX_ASSETS]

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} asset(s) ready** — platforms: {', '.join(selected_platforms)}")

    score_btn = st.button("⚡ Score Creatives", type="primary", disabled=not uploaded_files)

    if score_btn and uploaded_files:
        results = []
        progress = st.progress(0, text="Scoring assets…")

        for idx, uf in enumerate(uploaded_files):
            progress.progress((idx) / len(uploaded_files), text=f"Scoring {uf.name}…")

            img = Image.open(uf).convert("RGB")
            file_size_mb = uf.size / (1024 * 1024)
            ext = uf.name.rsplit(".", 1)[-1].lower() if "." in uf.name else "jpg"

            # Compliance per platform
            checks_by_platform = {}
            for platform in selected_platforms:
                checks_by_platform[platform] = run_compliance(platform, img, file_size_mb, ext)

            # AI scoring
            ai_result = None
            perf_score = None
            if api_key:
                try:
                    ai_result = score_with_ai(img, selected_platforms, audience_stage, market, api_key)
                    perf_score = compute_performance_score(ai_result)
                except Exception as e:
                    st.warning(f"AI scoring failed for {uf.name}: {e}")

            results.append({
                "name": uf.name,
                "img": img,
                "checks_by_platform": checks_by_platform,
                "ai_result": ai_result,
                "perf_score": perf_score,
            })

        progress.progress(1.0, text="Done!")

        # ── Batch summary ──────────────────────────────────────────────────
        if len(results) > 1:
            st.divider()
            st.subheader("📊 Batch Summary")
            import pandas as pd

            summary_rows = []
            for r in results:
                overall_compliance = "PASS"
                for checks in r["checks_by_platform"].values():
                    ov = overall_status(checks)
                    if ov == "FAIL":
                        overall_compliance = "FAIL"
                        break
                    if ov == "RISK":
                        overall_compliance = "RISK"

                v_label = "—"
                if r["perf_score"] is not None:
                    v_label, _ = verdict(r["perf_score"])

                summary_rows.append({
                    "Asset": r["name"],
                    "Performance Score": r["perf_score"] if r["perf_score"] is not None else "—",
                    "Verdict": v_label,
                    "Compliance": overall_compliance,
                })

            df = pd.DataFrame(summary_rows)
            if any(isinstance(r["perf_score"], float) for r in results):
                df = df.sort_values("Performance Score", ascending=False, na_position="last")
            st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Per-asset scorecards ───────────────────────────────────────────
        st.divider()
        st.subheader("📋 Asset Scorecards")
        for r in results:
            render_scorecard(
                r["name"], r["img"],
                r["checks_by_platform"], r["ai_result"],
                r["perf_score"], selected_platforms,
            )

    elif not uploaded_files:
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**✅ Compliance**\nFormat, size, resolution, aspect ratio, text density, safe zones")
        with col2:
            st.markdown("**🤖 AI Score**\n5 dimensions, 0–100 scale (requires OpenAI key)")
        with col3:
            st.markdown("**📊 Batch Summary**\nRanked table for multi-asset uploads")
        with col4:
            st.markdown("**📖 Legend**\nFull check reference in the sidebar")


if __name__ == "__main__":
    main()
