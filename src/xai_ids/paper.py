from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "outputs"
OUTPUT = ROOT / "output" / "pdf" / "xai_ids_ieee_paper.pdf"
METADATA_PATH = ROOT / "paper_metadata.yaml"


def paper_metadata() -> dict[str, str]:
    metadata = yaml.safe_load(METADATA_PATH.read_text(encoding="utf-8"))
    required = {"title", "authors", "affiliation", "date", "footer"}
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"Missing paper metadata fields: {sorted(missing)}")
    return {key: str(value) for key, value in metadata.items()}


def register_fonts() -> None:
    candidates = {
        "LiberationSerif": ROOT
        / ".venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSerif.ttf",
        "LiberationSerif-Bold": ROOT
        / ".venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSerif-Bold.ttf",
        "LiberationSerif-Italic": ROOT
        / ".venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSerif-Italic.ttf",
    }
    for name, path in candidates.items():
        if path.exists():
            pdfmetrics.registerFont(TTFont(name, str(path)))


def styles():
    register_fonts()
    serif = "LiberationSerif" if "LiberationSerif" in pdfmetrics.getRegisteredFontNames() else "Times-Roman"
    bold = "LiberationSerif-Bold" if "LiberationSerif-Bold" in pdfmetrics.getRegisteredFontNames() else "Times-Bold"
    italic = "LiberationSerif-Italic" if "LiberationSerif-Italic" in pdfmetrics.getRegisteredFontNames() else "Times-Italic"
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName=bold,
            fontSize=19,
            leading=21,
            alignment=TA_CENTER,
            spaceAfter=5,
            textColor=colors.black,
        ),
        "author": ParagraphStyle(
            "Author", fontName=serif, fontSize=10, leading=11, alignment=TA_CENTER, spaceAfter=6
        ),
        "abstract": ParagraphStyle(
            "Abstract", fontName=serif, fontSize=8.2, leading=9.4, alignment=TA_JUSTIFY, spaceAfter=4
        ),
        "body": ParagraphStyle(
            "Body",
            fontName=serif,
            fontSize=8.8,
            leading=10.0,
            alignment=TA_JUSTIFY,
            spaceAfter=3.8,
            allowWidows=0,
            allowOrphans=0,
        ),
        "section": ParagraphStyle(
            "Section",
            fontName=bold,
            fontSize=9.6,
            leading=10.8,
            alignment=TA_CENTER,
            spaceBefore=5,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "subsection": ParagraphStyle(
            "Subsection",
            fontName=italic,
            fontSize=9.2,
            leading=10.3,
            alignment=TA_LEFT,
            spaceBefore=3.5,
            spaceAfter=2,
            keepWithNext=True,
        ),
        "caption": ParagraphStyle(
            "Caption", fontName=serif, fontSize=7.5, leading=8.4, alignment=TA_JUSTIFY, spaceBefore=2, spaceAfter=5
        ),
        "table_caption": ParagraphStyle(
            "TableCaption", fontName=serif, fontSize=7.1, leading=8, alignment=TA_CENTER, spaceBefore=4, spaceAfter=2
        ),
        "reference": ParagraphStyle(
            "Reference", fontName=serif, fontSize=6.8, leading=7.35, alignment=TA_LEFT, leftIndent=10, firstLineIndent=-10, spaceAfter=1.0
        ),
        "equation": ParagraphStyle(
            "Equation", fontName=italic, fontSize=8.2, leading=10, alignment=TA_CENTER, spaceBefore=2, spaceAfter=4
        ),
    }


def header_footer(canvas, document):
    metadata = paper_metadata()
    canvas.saveState()
    canvas.setFont("Times-Roman", 6.5)
    canvas.drawString(0.55 * inch, 0.32 * inch, metadata["footer"])
    canvas.drawRightString(7.95 * inch, 0.32 * inch, str(document.page))
    canvas.restoreState()


def first_page(canvas, document):
    canvas.saveState()
    canvas.setLineWidth(0.4)
    canvas.line(0.55 * inch, 0.43 * inch, 7.95 * inch, 0.43 * inch)
    canvas.setFont("Times-Roman", 6.5)
    canvas.drawRightString(7.95 * inch, 0.27 * inch, str(document.page))
    canvas.restoreState()


def make_document(path: Path):
    metadata = paper_metadata()
    page_w, page_h = letter
    margin = 0.55 * inch
    gutter = 0.22 * inch
    usable = page_w - 2 * margin
    col_w = (usable - gutter) / 2
    # Reserve enough first-page width for the complete title, abstract, and
    # index terms. If this frame is too short, ReportLab flows the abstract
    # into the left body column before the explicit FrameBreak.
    title_bottom = 7.35 * inch
    title_height = page_h - title_bottom - 0.50 * inch
    first_body_top = 7.17 * inch
    body_bottom = 0.52 * inch
    first_body_height = first_body_top - body_bottom
    full_body_height = page_h - body_bottom - 0.54 * inch

    first_frames = [
        Frame(margin, title_bottom, usable, title_height, id="title", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
        Frame(margin, body_bottom, col_w, first_body_height, id="first_left", leftPadding=0, rightPadding=4, topPadding=0, bottomPadding=0),
        Frame(margin + col_w + gutter, body_bottom, col_w, first_body_height, id="first_right", leftPadding=4, rightPadding=0, topPadding=0, bottomPadding=0),
    ]
    later_frames = [
        Frame(margin, body_bottom, col_w, full_body_height, id="left", leftPadding=0, rightPadding=4, topPadding=0, bottomPadding=0),
        Frame(margin + col_w + gutter, body_bottom, col_w, full_body_height, id="right", leftPadding=4, rightPadding=0, topPadding=0, bottomPadding=0),
    ]
    document = BaseDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title=metadata["title"],
        author=metadata["authors"],
        subject="Explainable AI, intrusion detection, SHAP, LIME, fidelity, stability, latency",
    )
    document.addPageTemplates(
        [
            PageTemplate(id="First", frames=first_frames, onPage=first_page, pagesize=letter),
            PageTemplate(id="Later", frames=later_frames, onPage=header_footer, pagesize=letter),
        ]
    )
    return document, col_w


def para(text: str, style, **kwargs):
    return Paragraph(text, style, **kwargs)


def section(number: str, title: str, style):
    return Paragraph(f"{number}. {title.upper()}" if title else number, style)


def subsection(letter: str, title: str, style):
    return Paragraph(f"<i>{letter}. {title}</i>", style)


def make_table(data, widths, font_size=6.2, header=True, alignments=None):
    converted = []
    for row_index, row in enumerate(data):
        converted.append(
            [
                Paragraph(
                    str(value),
                    ParagraphStyle(
                        f"cell_{row_index}_{column_index}",
                        fontName="Times-Bold" if header and row_index == 0 else "Times-Roman",
                        fontSize=font_size,
                        leading=font_size + 1,
                        alignment=(alignments[column_index] if alignments else TA_CENTER),
                    ),
                )
                for column_index, value in enumerate(row)
            ]
        )
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="CENTER")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.1),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("LINEBELOW", (0, 0), (-1, 0), 0.65, colors.black),
        ("LINEABOVE", (0, 0), (-1, 0), 0.65, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.65, colors.black),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf3")))
    table.setStyle(TableStyle(commands))
    return table


def figure(path: Path, width: float, caption: str, number: int, style):
    image = Image(str(path))
    image._restrictSize(width, 3.0 * inch)
    return KeepTogether(
        [image, Paragraph(f"Fig. {number}. {caption}", style)]
    )


def predictive_table(frame: pd.DataFrame, dataset: str, col_w: float, s):
    summary = (
        frame[frame.dataset.eq(dataset)]
        .groupby(["task", "model"])
        .agg(
            accuracy=("accuracy", "mean"),
            balanced=("balanced_accuracy", "mean"),
            macro=("macro_f1", "mean"),
            macro_sd=("macro_f1", "std"),
        )
        .reset_index()
    )
    rows = [["Task", "Model", "Acc.", "Bal. acc.", "Macro-F1"]]
    for row in summary.itertuples():
        rows.append(
            [
                "Bin." if row.task == "binary" else "Multi.",
                "RF" if row.model == "random_forest" else "XGB",
                f"{row.accuracy:.3f}",
                f"{row.balanced:.3f}",
                f"{row.macro:.3f} +/- {row.macro_sd:.3f}",
            ]
        )
    return KeepTogether(
        [
            Paragraph(
                f"TABLE {'III' if dataset == 'CIC-IDS2017' else 'IV'}<br/>{dataset.upper()} PREDICTIVE RESULTS (FIVE SEEDS)",
                s["table_caption"],
            ),
            make_table(
                rows,
                [0.40 * inch, 0.36 * inch, 0.48 * inch, 0.58 * inch, col_w - 1.82 * inch],
                font_size=5.9,
            ),
        ]
    )


def xai_table(xai: pd.DataFrame, paired: pd.DataFrame, dataset: str, col_w: float, s):
    subset = xai[xai.dataset.eq(dataset)]
    rows = [["Task/model", "F-S", "F-L", "Delta [95% CI]", "p-H", "Med. ms S/L"]]
    for task in ("binary", "multiclass"):
        for model in ("random_forest", "xgboost"):
            pair = paired[
                paired.dataset.eq(dataset) & paired.task.eq(task) & paired.model.eq(model)
            ].iloc[0]
            group = subset[subset.task.eq(task) & subset.model.eq(model)].set_index("explainer")
            shap_row = group.loc["TreeSHAP"]
            lime_row = group.loc["LIME"]
            rows.append(
                [
                    f"{'Bin' if task == 'binary' else 'Multi'}/{'RF' if model == 'random_forest' else 'XGB'}",
                    f"{shap_row.fidelity_mean:.3f}",
                    f"{lime_row.fidelity_mean:.3f}",
                    f"{pair.mean_difference_shap_minus_lime:.3f} [{pair.ci_low:.3f}, {pair.ci_high:.3f}]",
                    "<0.0001" if pair.holm_adjusted_p < 0.0001 else f"{pair.holm_adjusted_p:.4f}",
                    f"{shap_row.latency_median_ms:.1f}/{lime_row.latency_median_ms:.1f}",
                ]
            )
    number = "V" if dataset == "CIC-IDS2017" else "VI"
    return KeepTogether(
        [
            Paragraph(
                f"TABLE {number}<br/>{dataset.upper()} EXPLANATION QUALITY AND LATENCY",
                s["table_caption"],
            ),
            make_table(
                rows,
                [0.48 * inch, 0.29 * inch, 0.29 * inch, 1.08 * inch, 0.38 * inch, col_w - 2.52 * inch],
                font_size=4.95,
            ),
        ]
    )


def dataset_protocol_table(col_w: float, s, manifest: dict):
    cic = manifest["datasets"]["CIC-IDS2017/binary"]
    unsw = manifest["datasets"]["UNSW-NB15/binary"]
    cic_test = round(cic["sampled_rows_after_deduplication"] * 0.20)
    cic_train = cic["sampled_rows_after_deduplication"] - cic_test
    rows = [
        ["Dataset", "Protocol", "Rows", "Feat.", "Classes"],
        ["CIC-IDS2017", "Capped, dedup.; strat. 80/20", f"{cic_train:,} / {cic_test:,}", "28", "2 / 15"],
        ["UNSW-NB15", "Predefined; dedup.; overlap-cleaned", f"{unsw['clean_training_rows']:,} / {unsw['clean_testing_rows']:,}", "34", "2 / 10"],
    ]
    return KeepTogether(
        [
            Paragraph("TABLE I<br/>DATASET AND EVALUATION PROTOCOL", s["table_caption"]),
            make_table(
                rows,
                [0.72 * inch, 1.18 * inch, 0.82 * inch, 0.35 * inch, col_w - 3.07 * inch],
                font_size=5.2,
            ),
        ]
    )


def experiment_settings_table(col_w: float, s, evaluation: dict):
    rows = [
        ["Component", "Fixed setting"],
        ["RF", "120 trees; depth 18; min. leaf 2; balanced bootstrap weights"],
        ["XGBoost", "160 trees; depth 7; eta 0.08; row/column subsample 0.85"],
        ["Explanations", f"{evaluation['explanation_samples']} matched flows/config.; LIME {evaluation['lime_samples']} samples; k={{1,3,5,10}}"],
        ["Inference", "2,000 paired bootstrap resamples; one-sided Wilcoxon; Holm correction"],
    ]
    return KeepTogether(
        [
            Paragraph("TABLE II<br/>PRESPECIFIED EXPERIMENT SETTINGS", s["table_caption"]),
            make_table(
                rows,
                [0.72 * inch, col_w - 0.72 * inch],
                font_size=5.5,
                alignments=[TA_LEFT, TA_LEFT],
            ),
        ]
    )


def ablation_table(ablation: pd.DataFrame, col_w: float, s):
    rows = [["Context", "Full k/F1", "Best reduced k/F1", "Delta"]]
    for dataset in ("CIC-IDS2017", "UNSW-NB15"):
        for task in ("binary", "multiclass"):
            for model in ("random_forest", "xgboost"):
                group = ablation[
                    ablation.dataset.eq(dataset)
                    & ablation.task.eq(task)
                    & ablation.model.eq(model)
                ].copy()
                full = group.loc[group.feature_count.idxmax()]
                reduced = group[group.feature_count.lt(full.feature_count)].sort_values("macro_f1").iloc[-1]
                label = (
                    f"{'CIC' if dataset == 'CIC-IDS2017' else 'UNSW'}/"
                    f"{'Bin' if task == 'binary' else 'Multi'}/"
                    f"{'RF' if model == 'random_forest' else 'XGB'}"
                )
                rows.append(
                    [
                        label,
                        f"{int(full.feature_count)}/{full.macro_f1:.3f}",
                        f"{int(reduced.feature_count)}/{reduced.macro_f1:.3f}",
                        f"{reduced.macro_f1 - full.macro_f1:+.3f}",
                    ]
                )
    return KeepTogether(
        [
            Paragraph("TABLE VII<br/>BEST REDUCED FEATURE SET VERSUS FULL MODEL", s["table_caption"]),
            make_table(
                rows,
                [0.78 * inch, 0.78 * inch, 1.15 * inch, col_w - 2.71 * inch],
                font_size=5.35,
            ),
        ]
    )


def robustness_table(
    baselines: pd.DataFrame,
    balancing: pd.DataFrame,
    calibration: pd.DataFrame,
    masking: pd.DataFrame,
    lime_budget: pd.DataFrame,
    col_w: float,
    s,
):
    baseline_range = baselines[baselines.baseline.eq("shallow_tree_depth5")].macro_f1
    balance = balancing.pivot_table(
        index=["dataset", "task", "model"], columns="training_weighting", values="macro_f1"
    )
    weighted_wins = int((balance["class_weighted"] > balance["none"]).sum())
    ece = calibration.expected_calibration_error
    mask_mean = masking.groupby("baseline").deletion_score.mean()
    lime = lime_budget.groupby("lime_samples").agg(
        stability=("mean_self_stability_cosine", "mean")
    )
    rows = [
        ["Audit", "Levels", "Main result"],
        ["Simple baseline", "Majority; depth-5 tree", f"Tree macro-F1 {baseline_range.min():.3f}-{baseline_range.max():.3f}"],
        ["Class weighting", "Weighted; none", f"Weighted higher in {weighted_wins}/8 contexts"],
        ["Calibration", "10 binary bins", f"ECE {ece.min():.3f}-{ece.max():.3f}"],
        ["Mask reference", "Median; zero; marginal; neighbors", f"Mean deletion {mask_mean.min():.3f}-{mask_mean.max():.3f}"],
        ["LIME budget", "100; 350; 700", f"Stability {lime.stability.iloc[0]:.3f}->{lime.stability.iloc[-1]:.3f}"],
    ]
    return KeepTogether(
        [
            Paragraph("TABLE VIII<br/>LITERATURE-DRIVEN ROBUSTNESS AUDITS", s["table_caption"]),
            make_table(rows, [0.25 * col_w, 0.31 * col_w, 0.44 * col_w], font_size=5.35),
        ]
    )


def build_story(col_w: float):
    s = styles()
    metadata = paper_metadata()
    predictive = pd.read_csv(RESULTS / "predictive_metrics.csv")
    per_class = pd.read_csv(RESULTS / "per_class_metrics.csv")
    xai = pd.read_csv(RESULTS / "explanation_summary.csv")
    paired = pd.read_csv(RESULTS / "paired_comparisons.csv")
    controls = pd.read_csv(RESULTS / "fidelity_control_tests.csv")
    ablation = pd.read_csv(RESULTS / "feature_ablation.csv")
    extended = RESULTS / "extended"
    predictive_baselines = pd.read_csv(extended / "predictive_baselines.csv")
    balancing = pd.read_csv(extended / "balancing_ablation.csv")
    calibration = pd.read_csv(extended / "calibration_summary.csv")
    masking = pd.read_csv(extended / "masking_baseline_ablation.csv")
    lime_budget = pd.read_csv(extended / "lime_budget_ablation.csv")
    with (RESULTS / "run_manifest.json").open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    evaluation = manifest["config"]["evaluation"]
    predictive_seed_count = len(manifest["config"]["predictive_seeds"])
    predictive_seed_values = ", ".join(str(seed) for seed in manifest["config"]["predictive_seeds"])
    explanation_count = int(evaluation["explanation_samples"])
    lime_repeat_count = int(evaluation.get("lime_repeat_seeds", 3))

    pred_summary = predictive.groupby(["dataset", "task", "model"]).mean(numeric_only=True)
    cic_xgb_bin = pred_summary.loc[("CIC-IDS2017", "binary", "xgboost")]
    cic_rf_multi = pred_summary.loc[("CIC-IDS2017", "multiclass", "random_forest")]
    cic_xgb_multi = pred_summary.loc[("CIC-IDS2017", "multiclass", "xgboost")]
    unsw_rf_bin = pred_summary.loc[("UNSW-NB15", "binary", "random_forest")]
    unsw_xgb_bin = pred_summary.loc[("UNSW-NB15", "binary", "xgboost")]
    unsw_rf_multi = pred_summary.loc[("UNSW-NB15", "multiclass", "random_forest")]
    unsw_xgb_multi = pred_summary.loc[("UNSW-NB15", "multiclass", "xgboost")]
    pair_min = paired.mean_difference_shap_minus_lime.min()
    pair_max = paired.mean_difference_shap_minus_lime.max()
    pair_p_max = paired.holm_adjusted_p.max()
    pair_dz_min = paired.paired_effect_dz.min()
    pair_dz_max = paired.paired_effect_dz.max()
    tree_passes = int(
        (controls.explainer.eq("TreeSHAP") & (controls.holm_adjusted_p < 0.05) & (controls.ci_low > 0)).sum()
    )
    lime_passes = int(
        (controls.explainer.eq("LIME") & (controls.holm_adjusted_p < 0.05) & (controls.ci_low > 0)).sum()
    )
    shap_stability = xai[xai.explainer.eq("TreeSHAP")].stability_cosine_mean
    lime_stability = xai[xai.explainer.eq("LIME")].stability_cosine_mean
    xgb_shap_latency = xai[xai.model.eq("xgboost") & xai.explainer.eq("TreeSHAP")].latency_median_ms
    xgb_lime_latency = xai[xai.model.eq("xgboost") & xai.explainer.eq("LIME")].latency_median_ms
    unsw_rf_multi_xai = xai[
        xai.dataset.eq("UNSW-NB15") & xai.task.eq("multiclass") & xai.model.eq("random_forest")
    ].set_index("explainer")
    mask_mean = masking.groupby("baseline").deletion_score.mean()
    lime_budget_summary = lime_budget.groupby("lime_samples").agg(
        fidelity=("mean_deletion_fidelity", "mean"),
        stability=("mean_self_stability_cosine", "mean"),
        latency=("median_latency_ms", "median"),
    )
    balance_comparison = balancing.pivot_table(
        index=["dataset", "task", "model"],
        columns="training_weighting",
        values="macro_f1",
    )
    weighted_wins = int(
        (balance_comparison["class_weighted"] > balance_comparison["none"]).sum()
    )
    failed_lime = controls[
        controls.explainer.eq("LIME")
        & ~((controls.holm_adjusted_p < 0.05) & (controls.ci_low > 0))
    ]
    failed_lime_labels = ", ".join(
        f"{'CIC' if row.dataset == 'CIC-IDS2017' else 'UNSW'} "
        f"{'binary' if row.task == 'binary' else 'multiclass'} "
        f"{'RF' if row.model == 'random_forest' else 'XGBoost'}"
        for row in failed_lime.itertuples()
    )
    unsw_metadata = manifest["datasets"]["UNSW-NB15/binary"]
    ablation_context = []
    for _, group in ablation.groupby(["dataset", "task", "model"]):
        full = group.loc[group.feature_count.idxmax()]
        reduced = group[group.feature_count.lt(full.feature_count)].sort_values("macro_f1").iloc[-1]
        ablation_context.append(float(reduced.macro_f1 - full.macro_f1))
    ablation_deltas = np.asarray(ablation_context)
    reduced_noninferior_count = int((ablation_deltas >= -0.005).sum())

    unsw_seed42_recall = per_class[
        per_class.dataset.eq("UNSW-NB15")
        & per_class.task.eq("multiclass")
        & per_class.seed.eq(42)
    ].pivot(index="class", columns="model", values="recall")
    minority_parts = []
    for attack_class in ("Analysis", "DoS", "Backdoor"):
        if attack_class in unsw_seed42_recall.index:
            row = unsw_seed42_recall.loc[attack_class]
            minority_parts.append(
                f"{attack_class} (RF {row['random_forest']:.3f}; XGBoost {row['xgboost']:.3f})"
            )
    minority_text = ", ".join(minority_parts)

    story = [
        Paragraph(
            metadata["title"],
            s["title"],
        ),
        Paragraph(
            f"{metadata['authors']}<br/><font size='8'>{metadata['affiliation']} - {metadata['date']}</font>",
            s["author"],
        ),
        Paragraph(
            f"<b>Abstract-</b> Machine-learning intrusion detection systems can be accurate while providing no defensible account of individual alerts. This work implements and evaluates an end-to-end benchmark that combines Random Forest (RF) and XGBoost classifiers with TreeSHAP and LIME explanations on CIC-IDS2017 and UNSW-NB15. Explanation quality is measured by controlled top-k deletion fidelity against random and bottom-ranked controls, nearest-neighbor stability, repeated-seed LIME self-stability, and wall-clock latency. Across {predictive_seed_count} model seeds, CIC binary macro-F1 reached "
            f"{cic_xgb_bin.macro_f1:.3f}; CIC multiclass macro-F1 was {cic_rf_multi.macro_f1:.3f}-{cic_xgb_multi.macro_f1:.3f}. On a leakage-controlled derivative of the predefined UNSW split, binary macro-F1 was {unsw_rf_bin.macro_f1:.3f}-{unsw_xgb_bin.macro_f1:.3f}, while multiclass macro-F1 was {unsw_xgb_multi.macro_f1:.3f}-{unsw_rf_multi.macro_f1:.3f}. TreeSHAP produced a larger mean confidence drop than LIME in all eight paired settings (Holm-adjusted one-sided p <= {pair_p_max:.2g}); paired differences ranged from {pair_min:.3f} to {pair_max:.3f}. TreeSHAP passed {tree_passes}/8 top-versus-random controls and LIME passed {lime_passes}/8. Alternative-reference and LIME-budget audits expose intervention and sampling sensitivity, while per-alert reports reject unsupported explanations. The framework therefore mitigates operational decision opacity without claiming causal or universal interpretability.",
            s["abstract"],
        ),
        Paragraph(
            "<b>Index Terms-</b> Explainable artificial intelligence, intrusion detection, SHAP, LIME, fidelity, stability, Random Forest, XGBoost.",
            s["abstract"],
        ),
        NextPageTemplate("Later"),
        FrameBreak(),
        section("I", "Introduction", s["section"]),
        para(
            "Modern anomaly-based intrusion detection systems (IDSs) can recognize attack patterns beyond fixed signatures, but their false alarms and opaque decisions complicate security operations [1]. A high-confidence alert is not automatically useful: an analyst must understand which flow characteristics drove the decision, whether the explanation is consistent with similar traffic, and whether it can be produced within an operational latency budget.",
            s["body"],
        ),
        para(
            "SHAP and LIME are widely used post-hoc explainers. SHAP assigns additive feature attributions based on Shapley-value principles [6], whereas LIME fits a sparse surrogate around one prediction [7]. Prior IDS studies commonly present attribution plots or use attributions for feature selection [8]-[10]. Visual plausibility, however, does not establish that the highlighted features actually control the model output. This work therefore treats explanation quality as a measurable system property.",
            s["body"],
        ),
        para(
            "This study contributes: (1) a leakage-conscious classifier-plus-explainer pipeline spanning two datasets, two task granularities, and two tree model families; (2) a quantitative XAI protocol that tests top-ranked features against random and bottom-ranked controls; (3) paired inference with bootstrap intervals, Wilcoxon tests, Holm correction, and standardized effects; (4) deployment guidance that jointly considers fidelity, stability, tail latency, minority classes, and feature count; and (5) an operational report that exposes each alert's signed evidence and accepts it only after global and alert-specific random-ranking controls.",
            s["body"],
        ),
        figure(
            RESULTS / "figures" / "paper_workflow.png",
            col_w,
            "End-to-end benchmark workflow. Training-only transformations and fixed held-out partitions precede matched explanation and quantitative validation.",
            1,
            s["caption"],
        ),
        section("II", "Related Work", s["section"]),
        para(
            "Khraisat <i>et al.</i> identify dataset age, false alarms, evasion, and inconsistent evaluation as persistent IDS problems [1]. CIC-IDS2017 was designed to add modern benign activity and diverse attacks across five capture days [2]. UNSW-NB15 contains nine attack families and provides an official 175,341/82,332 training/testing partition [3]. These datasets reduce dependence on legacy KDD-only evaluation, although neither is a direct sample of a production security operations center.",
            s["body"],
        ),
        para(
            "Mane and Rao demonstrated multiple explanation styles for a deep IDS but did not standardize quantitative validation [8]. Gaspar <i>et al.</i> tested whether LIME/SHAP perturbations could change IDS errors [9], while Ahmed <i>et al.</i> combined SHAP feature selection with CIC ensembles [10]. Alazab <i>et al.</i> used information gain to reduce NSL-KDD features [13], and Kasongo and Sun selected 19 UNSW features with XGBoost [14]; neither established a universal feature count. Yeh <i>et al.</i> argue that explanations require objective fidelity and sensitivity measures beyond plausibility [12]. This work joins these ideas through matched explainers, explicit controls, feature curves, robustness audits, latency distributions, and paired uncertainty.",
            s["body"],
        ),
        section("III", "Methodology", s["section"]),
        subsection("A", "Research Questions", s["subsection"]),
        para(
            "RQ1 asks how RF and XGBoost perform across datasets and task granularities. RQ2 asks whether deleting an explainer's top features reduces confidence more than deleting random or bottom-ranked features. RQ3 asks whether nearby same-prediction flows receive compatible explanations. RQ4 asks whether latency supports offline, analyst-on-demand, or near-real-time use. RQ5 asks whether conclusions persist across models, datasets, tasks, feature counts, and predictive seeds.",
            s["body"],
        ),
        subsection("B", "Data and Leakage Controls", s["subsection"]),
        para(
            f"The UNSW experiment begins from the predefined 175,341/82,332 training/testing files. Audit found within-partition duplication and exact feature vectors shared across files; exact rows were deduplicated and {unsw_metadata['cross_partition_feature_overlaps_removed_from_test']:,} overlapping test vectors were removed, leaving {unsw_metadata['clean_training_rows']:,}/{unsw_metadata['clean_testing_rows']:,} rows. The accessible CIC representation contains all 2,830,743 flows and 28 selected features but omits day and timestamp metadata. To control computation while retaining minority labels, at most 25,000 rows per label were sampled, 15,103 sampled duplicates were removed, and the remaining 124,513 rows were split 80/20 with label stratification. Neither dataset result is claimed as temporal generalization evidence.",
            s["body"],
        ),
        para(
            "All preprocessing is learned from training data. Infinite numeric values are converted to missing values and replaced with training medians. Categorical variables are ordinal-encoded with training-derived dictionaries and an explicit unseen category. Labels and attack categories are excluded from the feature space. Binary mappings use Normal versus Attack; multiclass experiments retain the supplied labels. After duplicate/overlap removal, the naturally imbalanced UNSW test distribution receives no class resampling or balancing.",
            s["body"],
        ),
        dataset_protocol_table(col_w, s, manifest),
        Spacer(1, 4),
        subsection("C", "Models and Predictive Measures", s["subsection"]),
        para(
            f"RF uses 120 trees, maximum depth 18, minimum leaf size 2, and balanced bootstrap class weights [4]. XGBoost uses 160 depth-7 trees, learning rate 0.08, 0.85 row/column subsampling, histogram training, and balanced sample weights [5]. Models are repeated with seeds {predictive_seed_values}. Accuracy, balanced accuracy, macro-F1, weighted-F1, per-class precision/recall/F1, confusion matrices, and binary PR-AUC, false-positive rate (FPR), and false-negative rate (FNR) are reported.",
            s["body"],
        ),
        experiment_settings_table(col_w, s, evaluation),
        Spacer(1, 4),
        subsection("D", "Explanation Metrics", s["subsection"]),
        para(
            f"For each configuration, {explanation_count} label-stratified test flows are explained by both methods, with available binary correct and error types explicitly represented. Attributions are converted to signed vectors in the original feature order. For an input <i>x</i>, predicted class <i>c</i>, feature ranking <i>r</i>, and training baseline <i>b</i>, the deletion score at <i>k</i> is",
            s["body"],
        ),
        para(
            "D<sub>k</sub>(x,r) = p(c|x) - p(c|mask(x, r<sub>1:k</sub>, b)).",
            s["equation"],
        ),
        para(
            "The primary fidelity score averages D<sub>k</sub> for k in {1,3,5,10}. The same calculation uses a fixed random ranking and the bottom attribution ranking as controls. Flip rate records whether masking changes the predicted class. Paired SHAP-minus-LIME differences are bootstrapped over shared instances with 2,000 resamples.",
            s["body"],
        ),
        para(
            f"Within each {explanation_count}-flow evaluation subset, neighborhood stability pairs each explained flow with its nearest standardized flow having the same predicted class, then reports signed cosine similarity and top-5 Jaccard overlap. LIME self-stability is estimated from {lime_repeat_count} seeds on {evaluation['self_stability_samples']} repeated flows. Explainer-call latency is measured per instance after one warm-up and summarized by median, P90, P95, P99, and throughput; it excludes data loading, preprocessing, prediction, serialization, and report rendering. LIME uses {evaluation['lime_samples']} perturbation samples; TreeSHAP uses the fitted tree model directly.",
            s["body"],
        ),
        para(
            "For explanation vectors a and a' assigned to nearest same-prediction flows, cosine stability is S<sub>cos</sub>=(a dot a')/(||a|| ||a'||); rank stability is the Jaccard overlap between their top-5 feature sets. These measures deliberately separate attribution magnitude/direction from rank agreement. A stable explanation is not necessarily faithful, so neither measure substitutes for deletion controls.",
            s["body"],
        ),
        subsection("E", "Statistical Analysis and Decision Rules", s["subsection"]),
        para(
            "Because TreeSHAP and LIME explain identical flows, the primary comparison uses within-flow differences in mean top-k deletion score. The mean difference and a percentile 95% confidence interval are obtained with 2,000 paired bootstrap resamples. The prespecified directional hypothesis (TreeSHAP fidelity greater than LIME) is tested with a one-sided Wilcoxon signed-rank test. Holm's step-down procedure controls family-wise error across the eight dataset-task-model comparisons. The same correction is applied to 16 top-versus-random control tests. Statistical significance requires adjusted p < 0.05; practical importance is reported as the raw probability-drop difference and paired effect d<sub>z</sub>.",
            s["body"],
        ),
        figure(
            RESULTS / "figures" / "predictive_performance.png",
            col_w,
            f"Macro-F1 by dataset, task, and model. Bars aggregate {predictive_seed_count} predictive seeds.",
            2,
            s["caption"],
        ),
        section("IV", "Experimental Results", s["section"]),
        subsection("A", "Predictive Performance", s["subsection"]),
        predictive_table(predictive, "CIC-IDS2017", col_w, s),
        Spacer(1, 4),
        para(
            f"CIC binary detection is strong: XGBoost achieves {cic_xgb_bin.accuracy:.4f} accuracy and {cic_xgb_bin.macro_f1:.4f} macro-F1 with FPR {cic_xgb_bin.false_positive_rate:.4f} and FNR {cic_xgb_bin.false_negative_rate:.4f}. Multiclass performance remains high but lower, with macro-F1 {cic_rf_multi.macro_f1:.4f} for RF and {cic_xgb_multi.macro_f1:.4f} for XGBoost. Seed standard deviations are at most 0.0036 macro-F1, indicating low training randomness relative to dataset effects.",
            s["body"],
        ),
        predictive_table(predictive, "UNSW-NB15", col_w, s),
        Spacer(1, 4),
        para(
            f"The leakage-controlled UNSW partition remains substantially harder. Binary macro-F1 is {unsw_rf_bin.macro_f1:.4f} (RF) and {unsw_xgb_bin.macro_f1:.4f} (XGBoost). XGBoost has FNR {unsw_xgb_bin.false_negative_rate:.4f} and FPR {unsw_xgb_bin.false_positive_rate:.4f}. Multiclass macro-F1 is {unsw_rf_multi.macro_f1:.4f} (RF) and {unsw_xgb_multi.macro_f1:.4f} (XGBoost). The macro/weighted gap shows that aggregate performance is influenced by frequent classes.",
            s["body"],
        ),
        subsection("B", "Explanation Fidelity and Stability", s["subsection"]),
        xai_table(xai, paired, "CIC-IDS2017", col_w, s),
        Spacer(1, 4),
        xai_table(xai, paired, "UNSW-NB15", col_w, s),
        Spacer(1, 4),
        para(
            f"TreeSHAP has higher median/mode-baseline top-k deletion fidelity than LIME in every configuration. Paired mean improvements range from {pair_min:.3f} to {pair_max:.3f}; all eight bootstrap intervals exclude zero, paired d<sub>z</sub> ranges from {pair_dz_min:.3f} to {pair_dz_max:.3f}, and every Holm-adjusted one-sided p-value is at most {pair_p_max:.2g}. Thus, the ranking comparison is consistent and statistically supported under the prespecified intervention.",
            s["body"],
        ),
        para(
            f"The control analysis is stricter. TreeSHAP's top-ranked features outperform random masking in {tree_passes}/8 configurations after Holm correction; LIME passes in {lime_passes}/8. The failed LIME context is {failed_lime_labels}; its ranking is not demonstrably better than random under the 350-sample perturbation budget.",
            s["body"],
        ),
        para(
            f"TreeSHAP's neighborhood cosine similarity ranges from {shap_stability.min():.3f} to {shap_stability.max():.3f}, compared with {lime_stability.min():.3f} to {lime_stability.max():.3f} for LIME. The findings reject the idea that a single attractive local bar chart is sufficient validation: an explainer can look plausible while failing a random deletion control or changing materially across nearby flows.",
            s["body"],
        ),
        figure(
            RESULTS / "figures" / "paper_paired_fidelity.png",
            col_w,
            "Paired TreeSHAP-minus-LIME fidelity differences. Every 95% bootstrap interval lies above zero; p-values in Tables V-VI are Holm-adjusted Wilcoxon results.",
            3,
            s["caption"],
        ),
        subsection("C", "Latency and Operational Fit", s["subsection"]),
        para(
            f"Latency depends on the model-explainer pair. For XGBoost, TreeSHAP median latency is {xgb_shap_latency.min():.1f}-{xgb_shap_latency.max():.1f} ms, compared with {xgb_lime_latency.min():.1f}-{xgb_lime_latency.max():.1f} ms for LIME. For UNSW RF multiclass, medians are {unsw_rf_multi_xai.loc['TreeSHAP'].latency_median_ms:.1f} ms for TreeSHAP and {unsw_rf_multi_xai.loc['LIME'].latency_median_ms:.1f} ms for LIME; TreeSHAP P95 is {unsw_rf_multi_xai.loc['TreeSHAP'].latency_p95_ms:.1f} ms. Deployment guidance therefore cannot be based on explainer name alone.",
            s["body"],
        ),
        figure(
            RESULTS / "figures" / "paper_latency_configurations.png",
            col_w,
            "Median single-flow latency for every model-explainer configuration on the fixed host. The horizontal axis is logarithmic.",
            4,
            s["caption"],
        ),
        subsection("D", "Feature-Count Ablation and Error Analysis", s["subsection"]),
        para(
            f"Feature reduction can preserve performance, but the useful count is context-specific. The best tested reduced set is within 0.005 macro-F1 of the full model in {reduced_noninferior_count}/8 contexts; reduced-minus-full changes range from {ablation_deltas.min():+.3f} to {ablation_deltas.max():+.3f}. Because these are seed-42 exploratory test-set comparisons rather than nested model selection, they motivate feature-count curves but do not establish a universally optimal k.",
            s["body"],
        ),
        ablation_table(ablation, col_w, s),
        Spacer(1, 4),
        figure(
            RESULTS / "figures" / "feature_ablation.png",
            col_w,
            "Feature-count ablation. Rankings come from the frozen full model; each reduced model is retrained with seed 42.",
            5,
            s["caption"],
        ),
        para(
            f"UNSW minority failures remain operationally important. Seed-42 recall for selected difficult classes is {minority_text}. CIC errors concentrate in Web Attack classes, but supports are small, so those rates have high sampling uncertainty. Global attributions are dataset-specific: CIC XGBoost repeatedly emphasizes Destination Port, backward packet-length variability, and initial TCP window bytes, whereas UNSW emphasizes load/byte features, service, and recent destination/source-port counts.",
            s["body"],
        ),
        figure(
            RESULTS / "figures" / "paper_unsw_recall.png",
            col_w,
            "UNSW-NB15 multiclass recall at seed 42. Analysis, Backdoor, and DoS remain weak despite acceptable weighted averages.",
            6,
            s["caption"],
        ),
        subsection("E", "Literature-Driven Robustness Audits", s["subsection"]),
        robustness_table(
            predictive_baselines, balancing, calibration, masking, lime_budget, col_w, s
        ),
        para(
            f"Both full model families exceed the majority and depth-5 tree baselines. With identical seed 42 fits, class weighting improves macro-F1 in {weighted_wins}/8 contexts, so balancing is not universally beneficial. Seed-42 binary expected calibration error is {calibration[calibration.dataset.eq('CIC-IDS2017')].expected_calibration_error.min():.3f}-{calibration[calibration.dataset.eq('CIC-IDS2017')].expected_calibration_error.max():.3f} on CIC and {calibration[calibration.dataset.eq('UNSW-NB15')].expected_calibration_error.min():.3f}-{calibration[calibration.dataset.eq('UNSW-NB15')].expected_calibration_error.max():.3f} on UNSW. On 12 flows per configuration, TreeSHAP deletion remains large with median/mode ({mask_mean['training_median_mode']:.3f}), zero ({mask_mean['zero']:.3f}), marginal-row ({mask_mean['marginal_training_row']:.3f}), and nearest-opposite ({mask_mean['nearest_opposite_prediction']:.3f}) references, but collapses for nearest same-prediction replacement ({mask_mean['nearest_same_prediction']:.3f}); fidelity is therefore an intervention-specific estimand. On six flows per configuration and five repeats, increasing LIME from 100 to 700 samples raises pooled self-stability from {lime_budget_summary.loc[100].stability:.3f} to {lime_budget_summary.loc[700].stability:.3f}, while median latency rises from {lime_budget_summary.loc[100].latency:.1f} to {lime_budget_summary.loc[700].latency:.1f} ms and fidelity changes little.",
            s["body"],
        ),
        section("V", "Discussion", s["section"]),
        subsection("A", "Answers to the Research Questions", s["subsection"]),
        para(
            "RQ1: RF and XGBoost beat simple baselines and are nearly tied on CIC and UNSW binary tasks, but neither solves UNSW multiclass detection. RQ2: TreeSHAP clears the random control in every configuration and exceeds LIME under the primary baseline, although absolute deletion depends on the reference intervention. RQ3: TreeSHAP is more neighbor-consistent within the selected subsets; LIME stability improves with sampling budget. RQ4: no explainer has a universal latency advantage. RQ5: the paired TreeSHAP ranking advantage persists across all tested contexts, while calibration, balancing, feature count, masking magnitude, and latency remain context-dependent.",
            s["body"],
        ),
        subsection("B", "Why the Explainers Differ", s["subsection"]),
        para(
            "TreeSHAP exploits the fitted tree structure and allocates credit consistently under its conditional-expectation assumptions. LIME instead samples a local neighborhood, discretizes features, and fits a surrogate. In mixed continuous/categorical flow spaces, that synthetic neighborhood may not match the data manifold; limited sampling also introduces variance. These mechanisms plausibly explain TreeSHAP's stronger fidelity and stability, but the experiment does not isolate which LIME component is responsible. Attribution magnitudes are not compared directly across methods because SHAP values and surrogate coefficients have different scales; only rankings and downstream behavioral tests are compared.",
            s["body"],
        ),
        subsection("C", "Operational Meaning", s["subsection"]),
        para(
            "The benchmark separates three questions that are often conflated. Predictive quality asks whether the IDS labels traffic correctly. Fidelity asks whether an explanation reflects the fitted model. Human usefulness asks whether an analyst can make a better decision. This work measures the first two and operational cost; it does not claim that a faithful explanation is automatically understandable, causal, or actionable. That distinction is central when using XAI in a security operations workflow.",
            s["body"],
        ),
        section("VI", "Deployment Recommendations", s["section"]),
        para(
            f"Classification should run on every flow, while explanation generation should be policy-triggered. XGBoost plus TreeSHAP is the strongest measured low-latency candidate because it combines high primary fidelity with {xgb_shap_latency.min():.1f}-{xgb_shap_latency.max():.1f} ms median explainer-call time. CIC RF plus TreeSHAP is suitable for analyst-on-demand use. UNSW RF TreeSHAP should be reserved for lower-volume investigation unless its {unsw_rf_multi_xai.loc['TreeSHAP'].latency_p95_ms:.1f} ms multiclass tail is acceptable. LIME can be faster than RF TreeSHAP, but {len(failed_lime)} tested ranking fails the random control; speed alone does not justify deployment.",
            s["body"],
        ),
        para(
            "The supplied transparency layer produces Markdown and JSON records containing predicted class and probability, top signed features and observed values, the training-derived masking baseline, local top/bottom/random deletion evidence, an empirical p-value, and benchmark stability and latency context. It labels evidence supported only when TreeSHAP passed the corrected global control and the alert clears a local randomization test; otherwise it requires review. This operationally mitigates decision opacity while explicitly warning that attribution is not causality. Governance should also monitor drift and class-specific fidelity.",
            s["body"],
        ),
        section("VII", "Threats to Validity", s["section"]),
        subsection("A", "Data and External Validity", s["subsection"]),
        para(
            "The CIC mirror contains a 28-feature preprocessing and no day metadata; therefore, its stratified split can still benefit from capture-specific artifacts and should not be interpreted as temporal generalization. Sampling was performed before deduplication to control memory, although duplicates were removed before splitting. UNSW uses a predefined partition derived from one testbed rather than a documented temporal holdout. Duplicate and cross-partition overlap removal reduces direct leakage but cannot make either benchmark representative of a live network.",
            s["body"],
        ),
        subsection("B", "Construct Validity", s["subsection"]),
        para(
            f"Masking can create implausible combinations because flow variables are correlated. The reference audit makes this dependence explicit: pooled top-k deletion is {mask_mean['training_median_mode']:.3f} for median/mode and {mask_mean['nearest_opposite_prediction']:.3f} for a nearest opposite prediction, but only {mask_mean['nearest_same_prediction']:.3f} for a nearest same prediction. These interventions answer different questions, and none proves causality. Cosine similarity may also penalize appropriate boundary changes. Most importantly, the study measures fitted-model faithfulness rather than human comprehension or improved analyst decisions.",
            s["body"],
        ),
        subsection("C", "Statistical Validity", s["subsection"]),
        para(
            f"Explanation inference uses {explanation_count} matched flows per configuration. Paired analysis and multiplicity correction improve efficiency and false-positive control, but intervals describe the selected flows rather than repeated datasets. The neighbor metric searches only within those selected flows and uses standardized Euclidean distance on a mixed encoded feature space. Feature-count results are exploratory because the reported best k is identified on the test results. Rare classes have small support. The {predictive_seed_count} predictive seeds quantify algorithmic variation, not uncertainty caused by alternative captures, preprocessing pipelines, or hyperparameter searches.",
            s["body"],
        ),
        subsection("D", "Scope", s["subsection"]),
        para(
            "Only tree classifiers and two post-hoc explainers are evaluated. The LIME budget audit covers 100-700 samples but not alternative kernels or discretizers. The calibration audit diagnoses raw probabilities but does not recalibrate them. SMOTE is omitted because interpolating encoded categorical flow fields can create invalid traffic; weighting versus no weighting is reported instead. Deep models, counterfactuals, calibrated thresholds, drift, adversarial manipulation, causal claims, and analyst studies remain outside scope.",
            s["body"],
        ),
        section("VIII", "Reproducibility", s["section"]),
        para(
            f"The run used Python {manifest['environment']['python']}, scikit-learn {manifest['environment']['scikit_learn']}, XGBoost {manifest['environment']['xgboost']}, and SHAP {manifest['environment']['shap']} on Windows 11 with {manifest['environment']['logical_cpu_count']} logical AMD64 CPUs. The release stores configuration, package pins, SHA-256 source hashes, row and deduplication audits, per-instance metrics, comparisons, plots, tests, and a machine-readable manifest. Raw data and serialized models are excluded from the public repository because of size and redistribution constraints; scripts regenerate them from locally supplied inputs.",
            s["body"],
        ),
        section("IX", "Conclusion", s["section"]),
        para(
            f"This study addresses black-box IDS decisions within a defined operational scope by evaluating detection and explanation as one system and producing rejectable per-alert evidence. TreeSHAP exceeds LIME in all eight primary paired comparisons, clears {tree_passes}/8 random-ranking controls, and is more neighborhood-stable; LIME clears {lime_passes}/8. Baseline, balancing, calibration, masking-reference, sampling-budget, class/error, feature-count, and latency audits expose where conclusions do and do not generalize. The transparency report converts this evidence into auditable alerts, but reference sensitivity confirms that post-hoc attribution is not causal or universal interpretability. Future work should add realistic conditional interventions, probability recalibration, drift tests, more explainers, and an analyst study.",
            s["body"],
        ),
        section("REFERENCES", "", s["section"]),
    ]

    references = [
        "[1] A. Khraisat, I. Gondal, P. Vamplew, and J. Kamruzzaman, \"Survey of intrusion detection systems: Techniques, datasets and challenges,\" <i>Cybersecurity</i>, vol. 2, art. no. 20, 2019, doi: 10.1186/s42400-019-0038-7.",
        "[2] I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, \"Toward generating a new intrusion detection dataset and intrusion traffic characterization,\" in <i>Proc. 4th Int. Conf. Information Systems Security and Privacy (ICISSP)</i>, 2018, pp. 108-116, doi: 10.5220/0006639801080116.",
        "[3] N. Moustafa and J. Slay, \"UNSW-NB15: A comprehensive data set for network intrusion detection systems,\" in <i>Proc. Military Communications and Information Systems Conf. (MilCIS)</i>, 2015, pp. 1-6, doi: 10.1109/MilCIS.2015.7348942.",
        "[4] L. Breiman, \"Random forests,\" <i>Machine Learning</i>, vol. 45, pp. 5-32, 2001, doi: 10.1023/A:1010933404324.",
        "[5] T. Chen and C. Guestrin, \"XGBoost: A scalable tree boosting system,\" in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining</i>, 2016, pp. 785-794, doi: 10.1145/2939672.2939785.",
        "[6] S. M. Lundberg and S.-I. Lee, \"A unified approach to interpreting model predictions,\" in <i>Advances in Neural Information Processing Systems</i>, vol. 30, 2017, pp. 4765-4774.",
        "[7] M. T. Ribeiro, S. Singh, and C. Guestrin, \"Why should I trust you?: Explaining the predictions of any classifier,\" in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining</i>, 2016, pp. 1135-1144, doi: 10.1145/2939672.2939778.",
        "[8] S. Mane and D. Rao, \"Explaining network intrusion detection system using explainable AI framework,\" <i>CoRR</i>, abs/2103.07110, 2021.",
        "[9] D. Gaspar, P. Silva, and C. Silva, \"Explainable AI for intrusion detection systems: LIME and SHAP applicability on multi-layer perceptron,\" <i>IEEE Access</i>, vol. 12, pp. 30164-30175, 2024, doi: 10.1109/ACCESS.2024.3368377.",
        "[10] U. Ahmed <i>et al.</i>, \"Hybrid bagging and boosting with SHAP based feature selection for enhanced predictive modeling in intrusion detection systems,\" <i>Scientific Reports</i>, vol. 14, art. no. 30532, 2024, doi: 10.1038/s41598-024-81151-1.",
        "[11] F. Pedregosa <i>et al.</i>, \"Scikit-learn: Machine learning in Python,\" <i>Journal of Machine Learning Research</i>, vol. 12, pp. 2825-2830, 2011.",
        "[12] C.-K. Yeh, C.-Y. Hsieh, A. Suggala, D. I. Inouye, and P. Ravikumar, \"On the (in)fidelity and sensitivity of explanations,\" in <i>Advances in Neural Information Processing Systems</i>, vol. 32, 2019.",
        "[13] A. Alazab, M. Hobbs, J. Abawajy, and M. Alazab, \"Using feature selection for intrusion detection system,\" in <i>Proc. Int. Symp. Communications and Information Technologies</i>, 2012, pp. 296-301, doi: 10.1109/ISCIT.2012.6380910.",
        "[14] S. M. Kasongo and Y. Sun, \"Performance analysis of intrusion detection systems using a feature selection method on the UNSW-NB15 dataset,\" <i>Journal of Big Data</i>, vol. 7, art. no. 105, 2020, doi: 10.1186/s40537-020-00379-6.",
    ]
    story.extend(Paragraph(reference, s["reference"]) for reference in references)
    return story


def main() -> None:
    from src.xai_ids.paper_figures import build_all

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    build_all(RESULTS)
    document, col_w = make_document(OUTPUT)
    document.build(build_story(col_w))
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
